#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from local_a3_agent_wrapper import SLICE_TYPES, normalize_slice_type


@dataclass(frozen=True)
class SliceTrainingScenario:
    case: str
    local_ues: int
    neighbor_ues: int
    local_load: float
    neighbor_load: float
    border_fraction: float
    local_radius: float = 190.0
    neighbor_radius: float = 230.0
    border_parallel_jitter: float = 45.0
    border_perp_jitter: float = 90.0


@dataclass(frozen=True)
class EpisodeTrainingScenario:
    name: str
    probability: float
    slices: Mapping[str, SliceTrainingScenario]

    def for_slice(self, slice_type: str) -> SliceTrainingScenario:
        normalized = normalize_slice_type(slice_type)
        if normalized in self.slices:
            return self.slices[normalized]
        return self.slices.get("default", NEUTRAL_SLICE_SCENARIO)


OFFLOAD_SLICE_SCENARIO = SliceTrainingScenario(
    case="offload",
    local_ues=4,
    neighbor_ues=1,
    local_load=0.90,
    neighbor_load=0.20,
    border_fraction=0.85,
    local_radius=160.0,
    neighbor_radius=220.0,
    border_parallel_jitter=30.0,
    border_perp_jitter=65.0,
)

NEUTRAL_SLICE_SCENARIO = SliceTrainingScenario(
    case="neutral",
    local_ues=3,
    neighbor_ues=2,
    local_load=0.60,
    neighbor_load=0.45,
    border_fraction=0.35,
)

RETAIN_SLICE_SCENARIO = SliceTrainingScenario(
    case="retain",
    local_ues=1,
    neighbor_ues=4,
    local_load=0.20,
    neighbor_load=0.85,
    border_fraction=0.10,
    local_radius=130.0,
    neighbor_radius=230.0,
)

RISKY_OFFLOAD_SLICE_SCENARIO = SliceTrainingScenario(
    case="risky_offload",
    local_ues=4,
    neighbor_ues=3,
    local_load=0.90,
    neighbor_load=0.82,
    border_fraction=0.70,
    local_radius=165.0,
    neighbor_radius=210.0,
    border_parallel_jitter=35.0,
    border_perp_jitter=75.0,
)


FEASIBLE_OFFLOAD_SLICE_SCENARIO = SliceTrainingScenario(
    case="offload",
    local_ues=4,
    neighbor_ues=1,
    local_load=0.88,
    neighbor_load=0.18,
    border_fraction=1.0,
    local_radius=155.0,
    neighbor_radius=215.0,
    border_parallel_jitter=8.0,
    border_perp_jitter=25.0,
)

FEASIBLE_RETAIN_SLICE_SCENARIO = SliceTrainingScenario(
    case="retain",
    local_ues=1,
    neighbor_ues=4,
    local_load=0.18,
    neighbor_load=0.88,
    border_fraction=0.05,
    local_radius=120.0,
    neighbor_radius=225.0,
    border_parallel_jitter=8.0,
    border_perp_jitter=25.0,
)

FEASIBLE_NEUTRAL_SLICE_SCENARIO = SliceTrainingScenario(
    case="neutral",
    local_ues=2,
    neighbor_ues=2,
    local_load=0.55,
    neighbor_load=0.50,
    border_fraction=0.20,
    local_radius=175.0,
    neighbor_radius=225.0,
    border_parallel_jitter=15.0,
    border_perp_jitter=35.0,
)


FIXED_EMBB_OFFLOAD_SLICE_SCENARIO = SliceTrainingScenario(
    case="offload",
    local_ues=4,
    neighbor_ues=1,
    local_load=0.90,
    neighbor_load=0.20,
    border_fraction=1.0,
    local_radius=160.0,
    neighbor_radius=220.0,
    border_parallel_jitter=20.0,
    border_perp_jitter=40.0,
)

FIXED_EMBB_RETAIN_SLICE_SCENARIO = SliceTrainingScenario(
    case="retain",
    local_ues=1,
    neighbor_ues=4,
    local_load=0.20,
    neighbor_load=0.90,
    border_fraction=0.25,
    local_radius=130.0,
    neighbor_radius=220.0,
    border_parallel_jitter=20.0,
    border_perp_jitter=40.0,
)

FIXED_EMBB_NEUTRAL_SLICE_SCENARIO = SliceTrainingScenario(
    case="neutral",
    local_ues=3,
    neighbor_ues=3,
    local_load=0.55,
    neighbor_load=0.50,
    border_fraction=0.40,
    local_radius=180.0,
    neighbor_radius=220.0,
    border_parallel_jitter=25.0,
    border_perp_jitter=50.0,
)


def _scenario(name: str, probability: float, **slice_cases: SliceTrainingScenario):
    normalized = {
        normalize_slice_type(slice_type): spec
        for slice_type, spec in slice_cases.items()
    }
    return EpisodeTrainingScenario(
        name=name,
        probability=float(probability),
        slices=normalized,
    )


FIXED_EMBB_OFFLOAD_SNAPSHOT = _scenario(
    "fixed_embb_offload_snapshot",
    1.0,
    eMBB=FIXED_EMBB_OFFLOAD_SLICE_SCENARIO,
)

FIXED_EMBB_RETAIN_SNAPSHOT = _scenario(
    "fixed_embb_retain_snapshot",
    1.0,
    eMBB=FIXED_EMBB_RETAIN_SLICE_SCENARIO,
)

FIXED_EMBB_NEUTRAL_SNAPSHOT = _scenario(
    "fixed_embb_neutral_snapshot",
    1.0,
    eMBB=FIXED_EMBB_NEUTRAL_SLICE_SCENARIO,
)


def fixed_embb_snapshot_scenarios(snapshot: str) -> tuple[EpisodeTrainingScenario, ...]:
    """Return fixed one-slice eMBB snapshots for local TD3 pretraining."""
    snapshot = str(snapshot).strip().lower()
    specs = {
        "offload": FIXED_EMBB_OFFLOAD_SNAPSHOT,
        "retain": FIXED_EMBB_RETAIN_SNAPSHOT,
        "neutral": FIXED_EMBB_NEUTRAL_SNAPSHOT,
    }
    if snapshot in specs:
        return (specs[snapshot],)
    if snapshot == "mixed":
        return tuple(
            EpisodeTrainingScenario(
                name=scenario.name,
                probability=1.0 / 3.0,
                slices=scenario.slices,
            )
            for scenario in (
                FIXED_EMBB_OFFLOAD_SNAPSHOT,
                FIXED_EMBB_RETAIN_SNAPSHOT,
                FIXED_EMBB_NEUTRAL_SNAPSHOT,
            )
        )
    raise ValueError(f"Unknown fixed eMBB snapshot mode: {snapshot!r}")


DEFAULT_LOCAL_A3_TRAINING_SCENARIOS = (
    _scenario(
        "embb_offload_urllc_retain_mmtc_neutral",
        0.18,
        eMBB=OFFLOAD_SLICE_SCENARIO,
        URLLC=RETAIN_SLICE_SCENARIO,
        mMTC=NEUTRAL_SLICE_SCENARIO,
    ),
    _scenario(
        "urllc_offload_embb_neutral_mmtc_retain",
        0.18,
        eMBB=NEUTRAL_SLICE_SCENARIO,
        URLLC=OFFLOAD_SLICE_SCENARIO,
        mMTC=RETAIN_SLICE_SCENARIO,
    ),
    _scenario(
        "mmtc_offload_embb_retain_urllc_neutral",
        0.18,
        eMBB=RETAIN_SLICE_SCENARIO,
        URLLC=NEUTRAL_SLICE_SCENARIO,
        mMTC=OFFLOAD_SLICE_SCENARIO,
    ),
    _scenario(
        "all_offload_clean",
        0.16,
        eMBB=OFFLOAD_SLICE_SCENARIO,
        URLLC=OFFLOAD_SLICE_SCENARIO,
        mMTC=OFFLOAD_SLICE_SCENARIO,
    ),
    _scenario(
        "all_retain",
        0.14,
        eMBB=RETAIN_SLICE_SCENARIO,
        URLLC=RETAIN_SLICE_SCENARIO,
        mMTC=RETAIN_SLICE_SCENARIO,
    ),
    _scenario(
        "risky_embb_urllc_mmtc_neutral",
        0.10,
        eMBB=RISKY_OFFLOAD_SLICE_SCENARIO,
        URLLC=RISKY_OFFLOAD_SLICE_SCENARIO,
        mMTC=NEUTRAL_SLICE_SCENARIO,
    ),
    _scenario(
        "all_neutral",
        0.06,
        eMBB=NEUTRAL_SLICE_SCENARIO,
        URLLC=NEUTRAL_SLICE_SCENARIO,
        mMTC=NEUTRAL_SLICE_SCENARIO,
    ),
)


FEASIBLE_MIXED_SLICE_SCENARIOS = (
    _scenario(
        "feasible_embb_offload_only",
        0.14,
        eMBB=FEASIBLE_OFFLOAD_SLICE_SCENARIO,
        URLLC=FEASIBLE_NEUTRAL_SLICE_SCENARIO,
        mMTC=FEASIBLE_NEUTRAL_SLICE_SCENARIO,
    ),
    _scenario(
        "feasible_urllc_offload_only",
        0.14,
        eMBB=FEASIBLE_NEUTRAL_SLICE_SCENARIO,
        URLLC=FEASIBLE_OFFLOAD_SLICE_SCENARIO,
        mMTC=FEASIBLE_NEUTRAL_SLICE_SCENARIO,
    ),
    _scenario(
        "feasible_mmtc_offload_only",
        0.14,
        eMBB=FEASIBLE_NEUTRAL_SLICE_SCENARIO,
        URLLC=FEASIBLE_NEUTRAL_SLICE_SCENARIO,
        mMTC=FEASIBLE_OFFLOAD_SLICE_SCENARIO,
    ),
    _scenario(
        "feasible_embb_offload_urllc_retain_mmtc_neutral",
        0.16,
        eMBB=FEASIBLE_OFFLOAD_SLICE_SCENARIO,
        URLLC=FEASIBLE_RETAIN_SLICE_SCENARIO,
        mMTC=FEASIBLE_NEUTRAL_SLICE_SCENARIO,
    ),
    _scenario(
        "feasible_urllc_offload_mmtc_retain_embb_neutral",
        0.16,
        eMBB=FEASIBLE_NEUTRAL_SLICE_SCENARIO,
        URLLC=FEASIBLE_OFFLOAD_SLICE_SCENARIO,
        mMTC=FEASIBLE_RETAIN_SLICE_SCENARIO,
    ),
    _scenario(
        "feasible_mmtc_offload_embb_retain_urllc_neutral",
        0.16,
        eMBB=FEASIBLE_RETAIN_SLICE_SCENARIO,
        URLLC=FEASIBLE_NEUTRAL_SLICE_SCENARIO,
        mMTC=FEASIBLE_OFFLOAD_SLICE_SCENARIO,
    ),
    _scenario(
        "feasible_all_offload",
        0.06,
        eMBB=FEASIBLE_OFFLOAD_SLICE_SCENARIO,
        URLLC=FEASIBLE_OFFLOAD_SLICE_SCENARIO,
        mMTC=FEASIBLE_OFFLOAD_SLICE_SCENARIO,
    ),
    _scenario(
        "feasible_all_neutral",
        0.04,
        eMBB=FEASIBLE_NEUTRAL_SLICE_SCENARIO,
        URLLC=FEASIBLE_NEUTRAL_SLICE_SCENARIO,
        mMTC=FEASIBLE_NEUTRAL_SLICE_SCENARIO,
    ),
)


def local_a3_training_scenario_set(name: str) -> tuple[EpisodeTrainingScenario, ...]:
    """Return a named local-A3 Stage-1 training scenario set."""
    normalized = str(name or "default").strip().lower().replace("-", "_")
    if normalized in {"default", "mixed", "default_mixed"}:
        return DEFAULT_LOCAL_A3_TRAINING_SCENARIOS
    if normalized in {"feasible", "feasible_mixed", "feasible_mixed_slices"}:
        return FEASIBLE_MIXED_SLICE_SCENARIOS
    raise ValueError(f"Unknown local A3 training scenario set: {name!r}")


def choose_training_scenario(
    rng: np.random.Generator,
    scenarios: Sequence[EpisodeTrainingScenario] = DEFAULT_LOCAL_A3_TRAINING_SCENARIOS,
) -> EpisodeTrainingScenario:
    if not scenarios:
        return EpisodeTrainingScenario(
            name="all_neutral",
            probability=1.0,
            slices={slice_type: NEUTRAL_SLICE_SCENARIO for slice_type in SLICE_TYPES},
        )

    probabilities = np.asarray(
        [max(float(scenario.probability), 0.0) for scenario in scenarios],
        dtype=float,
    )
    total = float(probabilities.sum())
    if total <= 0.0:
        probabilities = np.ones(len(scenarios), dtype=float) / len(scenarios)
    else:
        probabilities = probabilities / total

    idx = int(rng.choice(len(scenarios), p=probabilities))
    return scenarios[idx]
