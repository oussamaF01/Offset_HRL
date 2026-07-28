#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence, Tuple


def center_left_right_gnb_configs(
    center_gap_m: float,
    coverage_radius_m: float = 500.0,
) -> Tuple[dict, ...]:
    """Return a collinear left-center-right topology.

    Each gNB gets its own carrier_id. With a shared carrier, all 3 cells are
    mutual interferers (_compute_interference_watts sums real interference
    weighted by each neighbor's PRB duty factor), so simultaneously loading
    up multiple cells tanks SINR network-wide from genuine co-channel
    interference -- see PROJECT_SUMMARY.md Sec 3 for the same fix applied
    earlier for the same reason. Separating carriers removes that dimension
    entirely: SINR then reflects only distance/path-loss to the serving
    cell, not how busy neighboring cells are.
    """
    gap = float(center_gap_m)
    radius = float(coverage_radius_m)
    return tuple(
        {
            "id": gnb_id,
            "x": x,
            "y": 0.0,
            "coverage_radius": radius,
            "carrier_id": gnb_id,
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
    metadata: Mapping[str, object] = field(default_factory=dict)


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

# Three release tiers per side (medium_270m), instead of all 6 UEs sitting at
# the exact gNB1/outer midpoint. At the midpoint (x=±135, 0 dB signal
# difference) a single per-direction bias releases all same-side UEs at once
# -- there is no way to hold some back, so the only reachable end states are
# "all 6 stay" or "all 6 leave" (Jain = 1/3 or 2/3 on 3 gNBs, never 1.0).
# Shifting each side's 3 UEs by +/-15m from the midpoint gives each a
# different natural RSRP margin (verified against the actual link-budget
# model), so bias *magnitude* -- not just *sign* -- controls how many of the
# 3 release:
#   "easy"     (x = mid -15m, toward the outer gNB): margin +3.5 dB -> releases at bias < +0.41
#   "moderate" (x = +-135, the original exact midpoint): margin  0  dB -> releases at bias < -0.17
#   "hard"     (x = mid +15m, toward gNB1):          margin -3.5 dB -> releases at bias < -0.75
# A moderate per-direction bias (~-0.2..-0.3) now releases the easy+moderate
# pair on that side and leaves the hard one at gNB1, so applying it on both
# sides reaches a real 2-stay / 2-left / 2-right split -- Jain = 1.0 on 3
# gNBs, not just 2.
JAIN_CONTROL_6 = (
    (-150.0, -30.0),  # left, easy
    (135.0, -30.0),  # right, moderate
    (-135.0, 0.0),  # left, moderate
    (150.0, 0.0),  # right, easy
    (-120.0, 30.0),  # left, hard (stay-leaning)
    (120.0, 30.0),  # right, hard (stay-leaning)
)
JAIN_CONTROL_6_UPPER = tuple((x, y + 12.0) for x, y in JAIN_CONTROL_6)
JAIN_CONTROL_6_LOWER = tuple((x, y - 12.0) for x, y in JAIN_CONTROL_6)
# Outer-cell congestion case. In medium_270m, dx=142 m sits just beyond the
# gNB0/gNB1 and gNB2/gNB1 midpoint, so neutral A3 remains inactive while a
# moderate negative inward bias triggers handovers. Unlike the exact 135 m
# midpoint, this also stays inside all three gNB coverages in wide_320m.
OUTER_TO_CENTER_LEFT_6 = (
    (142.0, -20.0),
    (142.0, 0.0),
    (142.0, 20.0),
    (142.0, -20.0),
    (142.0, 0.0),
    (142.0, 20.0),
)
OUTER_TO_CENTER_RIGHT_6 = tuple((-x, y) for x, y in OUTER_TO_CENTER_LEFT_6)
# Same outer/center midpoints, shifted +12 m in y so a second slice placed
# here stays radio-equivalent to OUTER_TO_CENTER_*_6 while remaining a
# distinct UE placement (same convention as JAIN_CONTROL_6_UPPER).
OUTER_TO_CENTER_LEFT_6_UPPER = tuple((x, y + 12.0) for x, y in OUTER_TO_CENTER_LEFT_6)
OUTER_TO_CENTER_RIGHT_6_UPPER = tuple((x, y + 12.0) for x, y in OUTER_TO_CENTER_RIGHT_6)
OUTER_LEFT_OVERLAP_4 = (
    (160.0, -24.0),
    (160.0, 24.0),
    (185.0, -24.0),
    (185.0, 24.0),
)
OUTER_RIGHT_OVERLAP_4 = tuple((-x, y) for x, y in OUTER_LEFT_OVERLAP_4)


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
    UpperTrainingScenario(
        "jain_control_embb_urllc",
        1.0,
        (
            UpperUEGroup(
                "eMBB", 1, 6, 0.45,
                speed_mps=0.0,
                fixed_source_offsets_m=JAIN_CONTROL_6,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "URLLC", 1, 6, 0.35,
                speed_mps=0.0,
                fixed_source_offsets_m=JAIN_CONTROL_6_UPPER,
                placement_region="overlap",
            ),
        ),
        (
            "Two-slice controlled Jain scenario: eMBB and URLLC start on "
            "gNB1 with total demand above the safe gNB load, while both outer "
            "gNBs are light enough to accept released traffic."
        ),
        metadata={"slice_family": "two_slice", "feasible": True},
    ),
    UpperTrainingScenario(
        "jain_control_embb_mmtc",
        1.0,
        (
            UpperUEGroup(
                "eMBB", 1, 6, 0.45,
                speed_mps=0.0,
                fixed_source_offsets_m=JAIN_CONTROL_6,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "mMTC", 1, 6, 0.35,
                speed_mps=0.0,
                fixed_source_offsets_m=JAIN_CONTROL_6_LOWER,
                placement_region="overlap",
            ),
        ),
        (
            "Two-slice controlled Jain scenario: eMBB and mMTC start on "
            "gNB1. It keeps the same midpoint A3 controllability but uses "
            "only two active slices."
        ),
        metadata={"slice_family": "two_slice", "feasible": True},
    ),
    UpperTrainingScenario(
        "jain_control_urllc_mmtc",
        1.0,
        (
            UpperUEGroup(
                "URLLC", 1, 6, 0.40,
                speed_mps=0.0,
                fixed_source_offsets_m=JAIN_CONTROL_6_UPPER,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "mMTC", 1, 6, 0.35,
                speed_mps=0.0,
                fixed_source_offsets_m=JAIN_CONTROL_6_LOWER,
                placement_region="overlap",
            ),
        ),
        (
            "Two-slice controlled Jain scenario: URLLC and mMTC start on "
            "gNB1. The source gNB is overloaded in total, so the correct "
            "behavior is to release some demand toward lighter neighbors."
        ),
        metadata={"slice_family": "two_slice", "feasible": True},
    ),
    UpperTrainingScenario(
        "jain_control_outer_congested",
        1.0,
        (
            UpperUEGroup(
                "eMBB", 0, 6, 0.66,
                speed_mps=0.0,
                fixed_source_offsets_m=OUTER_TO_CENTER_LEFT_6,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "eMBB", 2, 6, 0.66,
                speed_mps=0.0,
                fixed_source_offsets_m=OUTER_TO_CENTER_RIGHT_6,
                placement_region="overlap",
            ),
        ),
        (
            "Outer-cell congestion scenario: eMBB demand starts on gNB0 and "
            "gNB2, each just above the 0.65 target, while gNB1 is empty. UEs "
            "are placed at each outer/center midpoint, so inward A3 handovers "
            "toward gNB1 are radio-feasible and safe admission can rebalance "
            "without creating an impossible target."
        ),
        metadata={"slice_family": "outer_congested", "feasible": True},
    ),
    UpperTrainingScenario(
        "jain_control_outer_congested_embb",
        1.0,
        (
            UpperUEGroup(
                "eMBB", 0, 6, 0.78,
                speed_mps=0.0,
                fixed_source_offsets_m=OUTER_TO_CENTER_LEFT_6,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "eMBB", 2, 6, 0.78,
                speed_mps=0.0,
                fixed_source_offsets_m=OUTER_TO_CENTER_RIGHT_6,
                placement_region="overlap",
            ),
        ),
        (
            "Moderately congested outer-cell scenario: eMBB demand starts on "
            "gNB0 and gNB2, each well above the 0.65 target (~0.78), while "
            "gNB1 is empty. Same inward-A3-feasible placement as "
            "jain_control_outer_congested but pushed further over target so "
            "relieving the congestion is unambiguous rather than marginal."
        ),
        metadata={"slice_family": "outer_congested", "feasible": True},
    ),
    UpperTrainingScenario(
        "jain_control_outer_congested_embb_urllc",
        1.0,
        (
            UpperUEGroup(
                "eMBB", 0, 6, 0.45,
                speed_mps=0.0,
                fixed_source_offsets_m=OUTER_TO_CENTER_LEFT_6,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "URLLC", 0, 6, 0.35,
                speed_mps=0.0,
                fixed_source_offsets_m=OUTER_TO_CENTER_LEFT_6_UPPER,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "eMBB", 2, 6, 0.45,
                speed_mps=0.0,
                fixed_source_offsets_m=OUTER_TO_CENTER_RIGHT_6,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "URLLC", 2, 6, 0.35,
                speed_mps=0.0,
                fixed_source_offsets_m=OUTER_TO_CENTER_RIGHT_6_UPPER,
                placement_region="overlap",
            ),
        ),
        (
            "Two-slice moderately congested outer-cell scenario: eMBB+URLLC "
            "demand starts on both gNB0 and gNB2 (~0.80 combined each), while "
            "gNB1 is empty. Inward A3 handovers toward gNB1 are radio-feasible "
            "and safe admission can rebalance both slices without creating an "
            "impossible target."
        ),
        metadata={"slice_family": "outer_congested", "feasible": True},
    ),
    UpperTrainingScenario(
        "jain_control_embb_urllc_impossible",
        1.0,
        (
            UpperUEGroup(
                "eMBB", 1, 6, 0.45,
                speed_mps=0.0,
                fixed_source_offsets_m=JAIN_CONTROL_6,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "URLLC", 1, 6, 0.35,
                speed_mps=0.0,
                fixed_source_offsets_m=JAIN_CONTROL_6_UPPER,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "eMBB", 0, 4, 0.35,
                speed_mps=0.0,
                fixed_source_offsets_m=OUTER_LEFT_OVERLAP_4,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "URLLC", 0, 4, 0.35,
                speed_mps=0.0,
                fixed_source_offsets_m=OUTER_LEFT_OVERLAP_4,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "eMBB", 2, 4, 0.35,
                speed_mps=0.0,
                fixed_source_offsets_m=OUTER_RIGHT_OVERLAP_4,
                placement_region="overlap",
            ),
            UpperUEGroup(
                "URLLC", 2, 4, 0.35,
                speed_mps=0.0,
                fixed_source_offsets_m=OUTER_RIGHT_OVERLAP_4,
                placement_region="overlap",
            ),
        ),
        (
            "Impossible two-slice controlled scenario: gNB1 is overloaded with "
            "eMBB+URLLC, but both outer gNBs are already above the safe total "
            "load. There is no light gNodeB target, so this case should not "
            "produce a useful outward migration."
        ),
        metadata={"slice_family": "two_slice", "feasible": False},
    ),
)

UPPER_TRAINING_SCENARIO_BY_NAME = {
    scenario.name: scenario for scenario in UPPER_TRAINING_SCENARIOS
}


def get_upper_training_scenarios(
    names=None,
) -> tuple[UpperTrainingScenario, ...]:
    if isinstance(names, Sequence) and not isinstance(names, str):
        if all(isinstance(item, UpperTrainingScenario) for item in names):
            return tuple(names)
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
