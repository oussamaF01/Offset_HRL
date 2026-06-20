#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


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


@dataclass(frozen=True)
class UpperTrainingScenario:
    name: str
    duration_s: float
    groups: Tuple[UpperUEGroup, ...]
    description: str
    tier: str = "slow"


UPPER_TRAINING_SCENARIOS = (
    UpperTrainingScenario(
        "fixed_embb_g0_overlap",
        16.0,
        (UpperUEGroup("eMBB", 0, 10, 0.86, 1, 0.0, path_progress=0.40, lateral_offset_m=45.0),),
        "Fixed eMBB UEs in the gNB0-gNB1 overlap teach release intensity.",
        "fixed",
    ),
    UpperTrainingScenario(
        "fixed_urllc_g1_overlap",
        16.0,
        (UpperUEGroup("URLLC", 1, 10, 0.84, 2, 0.0, path_progress=0.40, lateral_offset_m=-30.0),),
        "Fixed URLLC UEs in the gNB1-gNB2 overlap teach stable directional control.",
        "fixed",
    ),
    UpperTrainingScenario(
        "fixed_mmtc_g2_overlap",
        16.0,
        (UpperUEGroup("mMTC", 2, 12, 0.84, 0, 0.0, path_progress=0.40, lateral_offset_m=25.0),),
        "Fixed mMTC UEs in the gNB2-gNB0 overlap provide fine load granularity.",
        "fixed",
    ),
    UpperTrainingScenario(
        "fixed_dual_slice_g1",
        18.0,
        (
            UpperUEGroup("eMBB", 1, 10, 0.84, 0, 0.0, path_progress=0.40, lateral_offset_m=-40.0),
            UpperUEGroup("URLLC", 1, 10, 0.80, 2, 0.0, path_progress=0.40, lateral_offset_m=35.0),
        ),
        "Two fixed slices at gNB1 must be directed toward different neighbors.",
        "fixed",
    ),
    UpperTrainingScenario(
        "fixed_competing_sources_g1",
        18.0,
        (
            UpperUEGroup("eMBB", 0, 10, 0.84, 1, 0.0, path_progress=0.40, lateral_offset_m=45.0),
            UpperUEGroup("URLLC", 2, 10, 0.82, 1, 0.0, path_progress=0.40, lateral_offset_m=-25.0),
        ),
        "Two fixed overloaded sources compete for safe admission at gNB1.",
        "fixed",
    ),
    UpperTrainingScenario(
        "fixed_preloaded_target",
        18.0,
        (
            UpperUEGroup("eMBB", 0, 10, 0.88, 1, 0.0, path_progress=0.40, lateral_offset_m=45.0),
            UpperUEGroup("eMBB", 1, 8, 0.58),
        ),
        "Fixed overlap with a preloaded eMBB target teaches conservative offloading.",
        "fixed",
    ),
    UpperTrainingScenario(
        "embb_g0_to_g1_slow",
        20.0,
        (UpperUEGroup("eMBB", 0, 10, 0.90, 1, 3.0, path_progress=0.30, lateral_offset_m=60.0),),
        "Five eMBB UEs slowly move from overloaded gNB0 toward gNB1.",
    ),
    UpperTrainingScenario(
        "urllc_g1_to_g2_slow",
        20.0,
        (UpperUEGroup("URLLC", 1, 10, 0.88, 2, 3.0, path_progress=0.30, lateral_offset_m=-35.0),),
        "Four URLLC UEs move from overloaded gNB1 toward gNB2.",
    ),
    UpperTrainingScenario(
        "mmtc_g2_to_g0_slow",
        20.0,
        (UpperUEGroup("mMTC", 2, 12, 0.84, 0, 2.5, path_progress=0.30, lateral_offset_m=25.0),),
        "Six mMTC UEs move from overloaded gNB2 toward gNB0.",
    ),
    UpperTrainingScenario(
        "mixed_g0_release",
        24.0,
        (
            UpperUEGroup("eMBB", 0, 10, 0.90, 1, 3.0, path_progress=0.30, lateral_offset_m=50.0),
            UpperUEGroup("URLLC", 0, 10, 0.75, 1, 2.5, path_progress=0.30, lateral_offset_m=65.0),
            UpperUEGroup("mMTC", 0, 12, 0.60, 1, 2.0, path_progress=0.30, lateral_offset_m=80.0),
        ),
        "All three slices are overloaded at gNB0 and move toward gNB1.",
    ),
    UpperTrainingScenario(
        "three_way_slice_conflict",
        24.0,
        (
            UpperUEGroup("eMBB", 0, 10, 0.90, 1, 4.0, lateral_offset_m=45.0),
            UpperUEGroup("URLLC", 1, 10, 0.86, 2, 3.5, lateral_offset_m=-30.0),
            UpperUEGroup("mMTC", 2, 12, 0.82, 0, 3.0, lateral_offset_m=20.0),
        ),
        "Each gNB releases a different overloaded slice.",
        "fast",
    ),
    UpperTrainingScenario(
        "balanced_mixed_hold",
        12.0,
        tuple(
            UpperUEGroup(slice_type, gnb_id, 2, 0.42)
            for gnb_id in range(3)
            for slice_type in ("eMBB", "URLLC", "mMTC")
        ),
        "Balanced mixed traffic teaches neutral and retain behavior.",
        "fixed",
    ),
    UpperTrainingScenario(
        "embb_g1_to_g0_fast",
        28.0,
        (UpperUEGroup("eMBB", 1, 10, 0.88, 0, 6.0, path_progress=0.25, lateral_offset_m=-55.0),),
        "Fast reverse-direction eMBB release from gNB1 toward gNB0.",
        "fast",
    ),
    UpperTrainingScenario(
        "embb_g2_to_g1_moderate",
        30.0,
        (UpperUEGroup("eMBB", 2, 8, 0.72, 1, 3.0, path_progress=0.30, lateral_offset_m=30.0),),
        "Moderate eMBB congestion at gNB2 with a slower move toward gNB1.",
    ),
    UpperTrainingScenario(
        "urllc_g0_to_g2_fast",
        28.0,
        (UpperUEGroup("URLLC", 0, 10, 0.86, 2, 5.0, path_progress=0.25, lateral_offset_m=25.0),),
        "Fast URLLC movement from gNB0 toward gNB2.",
        "fast",
    ),
    UpperTrainingScenario(
        "urllc_g2_to_g1_moderate",
        30.0,
        (UpperUEGroup("URLLC", 2, 8, 0.70, 1, 2.5, path_progress=0.30, lateral_offset_m=-25.0),),
        "Moderate URLLC load moves from gNB2 toward gNB1.",
    ),
    UpperTrainingScenario(
        "mmtc_g0_to_g2_dense",
        32.0,
        (UpperUEGroup("mMTC", 0, 12, 0.88, 2, 2.0, path_progress=0.30, lateral_offset_m=35.0),),
        "Dense, slow mMTC population moves from gNB0 toward gNB2.",
    ),
    UpperTrainingScenario(
        "mmtc_g1_to_g0_moderate",
        30.0,
        (UpperUEGroup("mMTC", 1, 10, 0.68, 0, 2.5, path_progress=0.30, lateral_offset_m=-40.0),),
        "Moderate mMTC release from gNB1 toward gNB0.",
    ),
    UpperTrainingScenario(
        "dual_slice_g1_split_targets",
        30.0,
        (
            UpperUEGroup("eMBB", 1, 10, 0.86, 0, 3.0, path_progress=0.30, lateral_offset_m=-45.0),
            UpperUEGroup("URLLC", 1, 10, 0.78, 2, 2.5, path_progress=0.30, lateral_offset_m=35.0),
        ),
        "gNB1 releases eMBB toward gNB0 and URLLC toward gNB2.",
    ),
    UpperTrainingScenario(
        "dual_source_compete_for_g1",
        30.0,
        (
            UpperUEGroup("eMBB", 0, 10, 0.86, 1, 3.0, path_progress=0.30, lateral_offset_m=50.0),
            UpperUEGroup("URLLC", 2, 10, 0.82, 1, 2.5, path_progress=0.30, lateral_offset_m=-25.0),
        ),
        "Two overloaded sources compete for admission capacity at gNB1.",
    ),
    UpperTrainingScenario(
        "preloaded_target_embb",
        30.0,
        (
            UpperUEGroup("eMBB", 0, 10, 0.90, 1, 3.0, path_progress=0.30, lateral_offset_m=55.0),
            UpperUEGroup("eMBB", 1, 8, 0.62),
        ),
        "gNB0 wants to offload eMBB toward a target that is already moderately loaded.",
    ),
    UpperTrainingScenario(
        "preloaded_target_mixed",
        32.0,
        (
            UpperUEGroup("eMBB", 0, 10, 0.88, 1, 3.0, path_progress=0.30, lateral_offset_m=45.0),
            UpperUEGroup("URLLC", 2, 10, 0.82, 1, 2.5, path_progress=0.30, lateral_offset_m=-30.0),
            UpperUEGroup("mMTC", 1, 10, 0.58),
        ),
        "Mixed sources approach gNB1 while its mMTC slice is preloaded.",
    ),
    UpperTrainingScenario(
        "mixed_g2_release_reverse",
        32.0,
        (
            UpperUEGroup("eMBB", 2, 10, 0.88, 0, 3.0, path_progress=0.30, lateral_offset_m=25.0),
            UpperUEGroup("URLLC", 2, 10, 0.74, 0, 2.5, path_progress=0.30, lateral_offset_m=40.0),
            UpperUEGroup("mMTC", 2, 12, 0.66, 0, 2.0, path_progress=0.30, lateral_offset_m=55.0),
        ),
        "All slices release from gNB2 toward gNB0.",
    ),
    UpperTrainingScenario(
        "near_balanced_small_perturbation",
        16.0,
        (
            UpperUEGroup("eMBB", 0, 2, 0.52),
            UpperUEGroup("eMBB", 1, 2, 0.46),
            UpperUEGroup("eMBB", 2, 2, 0.42),
            UpperUEGroup("URLLC", 0, 2, 0.43),
            UpperUEGroup("URLLC", 1, 2, 0.51),
            UpperUEGroup("URLLC", 2, 2, 0.46),
            UpperUEGroup("mMTC", 0, 2, 0.45),
            UpperUEGroup("mMTC", 1, 2, 0.42),
            UpperUEGroup("mMTC", 2, 2, 0.50),
        ),
        "Small near-balanced differences teach restrained, non-saturated biases.",
        "fixed",
    ),
    UpperTrainingScenario(
        "fast_border_crossing_embb",
        30.0,
        (UpperUEGroup("eMBB", 0, 10, 0.82, 1, 8.0, path_progress=0.32, lateral_offset_m=35.0),),
        "Fast eMBB border crossing tests reaction speed and handover stability.",
        "fast",
    ),
)

UPPER_TRAINING_SCENARIO_BY_NAME = {
    scenario.name: scenario for scenario in UPPER_TRAINING_SCENARIOS
}


def get_upper_training_scenarios(names=None) -> tuple[UpperTrainingScenario, ...]:
    if names is None or names == "all":
        return UPPER_TRAINING_SCENARIOS
    requested = (
        tuple(part.strip() for part in names.split(",") if part.strip())
        if isinstance(names, str)
        else tuple(names)
    )
    unknown = [name for name in requested if name not in UPPER_TRAINING_SCENARIO_BY_NAME]
    if unknown:
        known = ", ".join(UPPER_TRAINING_SCENARIO_BY_NAME)
        raise ValueError(f"Unknown upper scenarios {unknown}. Known: {known}")
    return tuple(UPPER_TRAINING_SCENARIO_BY_NAME[name] for name in requested)
