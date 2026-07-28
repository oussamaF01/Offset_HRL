#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from local_a3_agent_wrapper import SLICE_TYPES, normalize_slice_type
from multi_gnb_wrapper import DEFAULT_UE_TRAFFIC_PROFILES
from upper_agent_training_scenarios import UpperTrainingScenario, UpperUEGroup


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


RANDOM_DEMAND_SLICE_TYPES = ("eMBB", "URLLC", "mMTC")
RANDOM_DEMAND_CENTER_OFFSETS = (
    (-135.0, -30.0),
    (135.0, -30.0),
    (-135.0, 0.0),
    (135.0, 0.0),
    (-135.0, 30.0),
    (135.0, 30.0),
)
RANDOM_DEMAND_CENTER_OFFSETS_UPPER = tuple((x, y + 12.0) for x, y in RANDOM_DEMAND_CENTER_OFFSETS)
RANDOM_DEMAND_CENTER_OFFSETS_LOWER = tuple((x, y - 12.0) for x, y in RANDOM_DEMAND_CENTER_OFFSETS)
RANDOM_DEMAND_CENTER_OFFSETS_BY_SLICE = {
    "eMBB": RANDOM_DEMAND_CENTER_OFFSETS,
    "URLLC": RANDOM_DEMAND_CENTER_OFFSETS_UPPER,
    "mMTC": RANDOM_DEMAND_CENTER_OFFSETS_LOWER,
}


def _randomized_traffic_profiles(
    rng: np.random.Generator,
    *,
    enabled: bool = True,
    bit_rate_scale_range: tuple[float, float] = (0.5, 2.5),
    packet_size_scale_range: tuple[float, float] = (0.75, 1.5),
) -> dict:
    profiles = {}
    for slice_type, base in DEFAULT_UE_TRAFFIC_PROFILES.items():
        profile = dict(base)
        if enabled:
            profile["bit_rate"] = float(base["bit_rate"] * rng.uniform(*bit_rate_scale_range))
            profile["packet_size_bits"] = float(base["packet_size_bits"] * rng.uniform(*packet_size_scale_range))
        profiles[normalize_slice_type(slice_type)] = profile
    return profiles


def _controllable_offsets_for_slice(
    rng: np.random.Generator,
    slice_type: str,
    *,
    count: int = 6,
    placement_jitter_m: float = 18.0,
):
    base = RANDOM_DEMAND_CENTER_OFFSETS_BY_SLICE[normalize_slice_type(slice_type)]
    offsets = []
    for idx in range(max(int(count), 1)):
        x, y = base[idx % len(base)]
        x_jitter = float(np.clip(rng.normal(0.0, 0.45 * placement_jitter_m), -placement_jitter_m, placement_jitter_m))
        y_jitter = float(np.clip(rng.normal(0.0, placement_jitter_m), -2.0 * placement_jitter_m, 2.0 * placement_jitter_m))
        offsets.append((float(x + x_jitter), float(y + y_jitter)))
    return tuple(offsets)


@dataclass(frozen=True)
class RandomDemandScenario:
    name: str
    demand_matrix: np.ndarray
    case_type: str
    ue_counts: np.ndarray
    duration_s: float = 1.0
    metadata: Mapping[str, object] = field(default_factory=dict)




def _ue_count_for_demand_load(load: float) -> int:
    load = float(load)
    if load <= 1e-6:
        return 0
    if load >= 0.65:
        return 6
    if load >= 0.35:
        return 4
    return 2


def _demand_hash(matrix: np.ndarray) -> str:
    rounded = np.round(np.asarray(matrix, dtype=float), 3)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:10]


def demand_profile_records_from_matrix(
    demand_matrix: np.ndarray,
    *,
    ue_counts: np.ndarray | None = None,
    slice_types=RANDOM_DEMAND_SLICE_TYPES,
    prb_budget: int = 100,
) -> tuple[dict, ...]:
    """Serializable version of the existing demand profile keyed by gNB/slice."""
    D = np.asarray(demand_matrix, dtype=float)
    counts = (
        np.asarray(ue_counts, dtype=int)
        if ue_counts is not None
        else np.vectorize(_ue_count_for_demand_load)(D)
    )
    records = []
    for gnb_id in range(D.shape[0]):
        for s_idx, slice_type in enumerate(slice_types):
            target_load = float(D[gnb_id, s_idx])
            records.append({
                "gnb_id": int(gnb_id),
                "slice_type": normalize_slice_type(slice_type),
                "target_load": target_load,
                "demand_prb": float(round(target_load * int(prb_budget))),
                "offered_bits": 0.0,
                "n_ues": int(counts[gnb_id, s_idx]),
            })
    return tuple(records)


def _clip_demand(matrix, min_demand_load: float, max_demand_load: float):
    return np.clip(np.asarray(matrix, dtype=float), float(min_demand_load), float(max_demand_load))


def _enforce_min_gnb_load(
    D: np.ndarray,
    min_total_gnb_load: float,
    max_demand_load: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Boost underloaded gNBs so every gNB total load >= min_total_gnb_load."""
    D = D.copy()
    min_load = float(min_total_gnb_load)
    max_load = float(max_demand_load)
    for g in range(D.shape[0]):
        deficit = min_load - float(D[g].sum())
        if deficit <= 1e-9:
            continue
        for s in rng.permutation(D.shape[1]):
            headroom = max_load - float(D[g, s])
            if headroom <= 1e-9:
                continue
            boost = min(deficit, headroom)
            D[g, s] += boost
            deficit -= boost
            if deficit <= 1e-9:
                break
    return D


_SLICE_NAME_TO_IDX = {"embb": 0, "urllc": 1, "mmtc": 2}


def _sample_demand_matrix_for_case(
    rng: np.random.Generator,
    case_type: str,
    *,
    min_demand_load: float,
    max_demand_load: float,
    safe_total_gnb_load: float,
    min_total_gnb_load: float = 0.0,
):
    """Sample a 3×3 demand matrix (gNBs × slices) for a named scenario.

    Case-type naming convention
    ---------------------------
    g{n}_embb_overload   : gNB n eMBB heavily loaded; neighbors have eMBB headroom → offload
    g{n}_urllc_overload  : same for URLLC
    g{n}_mmtc_overload   : same for mMTC
    g{n}_total_overload  : all slices of gNB n over safe_total; one neighbor light → offload
    g{n}_no_target       : gNB n overloaded on one slice; ALL neighbors also full → retain
    balanced             : all gNBs similar mid-range load → neutral
    mixed_random         : fully uniform random → mixed actions
    """
    low  = float(min_demand_load)
    high = float(max_demand_load)

    # Default: all cells low — each branch overrides only what it needs.
    D = rng.uniform(low, min(high, 0.18), size=(3, 3))

    # ── balanced ──────────────────────────────────────────────────────────────
    if case_type == "balanced":
        D = rng.uniform(max(low, 0.40), min(high, 0.65), size=(3, 3))

    # ── g{n}_{slice}_overload — single slice on one gNB ──────────────────────
    elif "_overload" in case_type and "total" not in case_type:
        # parse "g{n}_{slice}_overload"
        parts   = case_type.split("_")           # ["g0", "embb", "overload"]
        g       = int(parts[0][1:])
        s       = _SLICE_NAME_TO_IDX[parts[1]]
        others  = [x for x in range(3) if x != g]
        light, heavy = (others[0], others[1]) if rng.random() < 0.5 else (others[1], others[0])
        # gNB g: only slice s is overloaded; other slices stay low
        D[g, s] = rng.uniform(max(low, 0.75), high)
        # light neighbor has clear headroom on slice s → feasible offload target
        D[light, s] = rng.uniform(low, min(high, 0.18))
        # heavy neighbor has some load on slice s (still under safe)
        D[heavy, s] = rng.uniform(low, min(high, 0.45))

    # ── g{n}_total_overload — all slices of one gNB over safe total ───────────
    elif "total_overload" in case_type:
        g      = int(case_type.split("_")[0][1:])
        others = [x for x in range(3) if x != g]
        light, heavy = (others[0], others[1]) if rng.random() < 0.5 else (others[1], others[0])
        # gNB g: distribute load across all slices so total > safe_total_gnb_load
        D[g, :] = rng.uniform(0.24, min(high, 0.38), size=3)
        if D[g, :].sum() <= float(safe_total_gnb_load):
            D[g, int(rng.integers(3))] += float(safe_total_gnb_load) - D[g, :].sum() + rng.uniform(0.08, 0.18)
        # light neighbor: all slices low → can absorb from any slice
        D[light, :] = rng.uniform(low, min(high, 0.22), size=3)
        # heavy neighbor: moderate load (might absorb one slice)
        D[heavy, :] = rng.uniform(low, min(high, 0.42), size=3)

    # ── g{n}_no_target — gNB n overloaded but nowhere to offload ─────────────
    elif "no_target" in case_type:
        g      = int(case_type.split("_")[0][1:])
        s      = int(rng.integers(3))            # random slice to overload
        others = [x for x in range(3) if x != g]
        D[g, s] = rng.uniform(max(low, 0.75), high)
        # both neighbors also heavily loaded on slice s → no feasible target
        for nb in others:
            D[nb, s] = rng.uniform(max(low, 0.72), high)

    # ── mixed_random ──────────────────────────────────────────────────────────
    elif case_type == "mixed_random":
        D = rng.uniform(low, high, size=(3, 3))

    else:
        raise ValueError(f"Unknown random demand case_type={case_type!r}")

    D = _clip_demand(D, low, high)
    if min_total_gnb_load > 0.0:
        D = _enforce_min_gnb_load(D, min_total_gnb_load, high, rng)
    return D


# All structured case types used by generate_demand_curriculum.
# Named as  g{gnb}_{slice}_overload  so the expected expert action is self-evident.
DEMAND_CASE_TYPES: tuple[str, ...] = (
    # ── single-slice overload — one gNB, one slice, clear offload target ──────
    "g0_embb_overload",  "g0_urllc_overload",  "g0_mmtc_overload",
    "g1_embb_overload",  "g1_urllc_overload",  "g1_mmtc_overload",
    "g2_embb_overload",  "g2_urllc_overload",  "g2_mmtc_overload",
    # ── total overload — all slices of one gNB over safe threshold ────────────
    "g0_total_overload", "g1_total_overload",  "g2_total_overload",
    # ── no feasible target — overloaded but neighbors also full → retain ───────
    "g0_no_target",      "g1_no_target",       "g2_no_target",
    # ── reference baselines ───────────────────────────────────────────────────
    "balanced",
    "mixed_random",
)


def _upper_groups_from_demand_matrix(
    demand_matrix: np.ndarray,
    *,
    slice_types=RANDOM_DEMAND_SLICE_TYPES,
    rng: np.random.Generator | None = None,
    placement_jitter_m: float = 18.0,
):
    rng = np.random.default_rng() if rng is None else rng
    groups = []
    ue_counts = np.zeros_like(np.asarray(demand_matrix, dtype=float), dtype=int)
    for gnb_id in range(3):
        for s_idx, slice_type in enumerate(slice_types):
            load = float(demand_matrix[gnb_id, s_idx])
            count = _ue_count_for_demand_load(load)
            ue_counts[gnb_id, s_idx] = count
            if count <= 0:
                continue
            kwargs = {}
            if int(gnb_id) == 1:
                kwargs["placement_target_gnbs"] = (0, 2)
                kwargs["fixed_source_offsets_m"] = _controllable_offsets_for_slice(
                    rng,
                    slice_type,
                    count=max(count, 6),
                    placement_jitter_m=placement_jitter_m,
                )
            else:
                kwargs["target_gnb"] = 1
                kwargs["path_progress"] = float(np.clip(rng.normal(0.50, 0.035), 0.42, 0.58))
                kwargs["lateral_offset_m"] = float((s_idx - 1) * 14.0 + rng.normal(0.0, placement_jitter_m))
            groups.append(
                UpperUEGroup(
                    normalize_slice_type(slice_type),
                    int(gnb_id),
                    int(count),
                    float(load),
                    speed_mps=0.0,
                    placement_region="overlap",
                    **kwargs,
                )
            )
    return tuple(groups), ue_counts


def generate_random_demand_scenario(
    *,
    rng: np.random.Generator | None = None,
    index: int = 0,
    case_type: str = "mixed_random",
    min_demand_load: float = 0.05,
    max_demand_load: float = 0.95,
    safe_total_gnb_load: float = 0.65,
    min_total_gnb_load: float = 0.25,
    duration_s: float = 1.0,
    randomize_traffic_profiles: bool = True,
    bit_rate_scale_range: tuple[float, float] = (0.5, 2.5),
    packet_size_scale_range: tuple[float, float] = (0.75, 1.5),
    placement_jitter_m: float = 18.0,
) -> UpperTrainingScenario:
    rng = np.random.default_rng() if rng is None else rng
    D = _sample_demand_matrix_for_case(
        rng,
        case_type,
        min_demand_load=min_demand_load,
        max_demand_load=max_demand_load,
        safe_total_gnb_load=safe_total_gnb_load,
        min_total_gnb_load=min_total_gnb_load,
    )
    groups, ue_counts = _upper_groups_from_demand_matrix(
        D,
        rng=rng,
        placement_jitter_m=placement_jitter_m,
    )
    traffic_profiles = _randomized_traffic_profiles(
        rng,
        enabled=randomize_traffic_profiles,
        bit_rate_scale_range=bit_rate_scale_range,
        packet_size_scale_range=packet_size_scale_range,
    )
    digest = _demand_hash(D)
    name = f"random_demand_{index:04d}_{case_type}_{digest}"
    total_load = D.sum(axis=1)
    metadata = {
        "case_type": case_type,
        "demand_matrix": np.round(D, 4).tolist(),
        "total_demand_load": np.round(total_load, 4).tolist(),
        "ue_counts": ue_counts.tolist(),
        "demand_profile": tuple(demand_profile_records_from_matrix(D, ue_counts=ue_counts)),
        "traffic_profiles": traffic_profiles,
        "placement_jitter_m": float(placement_jitter_m),
        "demand_hash": digest,
        "safe_total_gnb_load": float(safe_total_gnb_load),
    }
    return UpperTrainingScenario(
        name=name,
        duration_s=float(duration_s),
        groups=groups,
        description=f"Random static demand scenario ({case_type}) with fixed per-episode demand matrix.",
        tier="random_demand",
        metadata=metadata,
    )


def generate_demand_curriculum(
    *,
    num_scenarios: int = 256,
    demand_seed: int = 7,
    min_demand_load: float = 0.05,
    max_demand_load: float = 0.95,
    safe_total_gnb_load: float = 0.65,
    min_total_gnb_load: float = 0.25,
    duration_s: float = 1.0,
    randomize_traffic_profiles: bool = True,
    bit_rate_scale_range: tuple[float, float] = (0.5, 2.5),
    packet_size_scale_range: tuple[float, float] = (0.75, 1.5),
    placement_jitter_m: float = 18.0,
) -> tuple[UpperTrainingScenario, ...]:
    rng = np.random.default_rng(int(demand_seed))
    scenarios = []
    for idx in range(max(1, int(num_scenarios))):
        case_type = DEMAND_CASE_TYPES[idx % len(DEMAND_CASE_TYPES)]
        scenarios.append(
            generate_random_demand_scenario(
                rng=rng,
                index=idx,
                case_type=case_type,
                min_demand_load=min_demand_load,
                max_demand_load=max_demand_load,
                safe_total_gnb_load=safe_total_gnb_load,
                min_total_gnb_load=min_total_gnb_load,
                duration_s=duration_s,
                randomize_traffic_profiles=randomize_traffic_profiles,
                bit_rate_scale_range=bit_rate_scale_range,
                packet_size_scale_range=packet_size_scale_range,
                placement_jitter_m=placement_jitter_m,
            )
        )
    return tuple(scenarios)


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
