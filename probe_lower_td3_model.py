#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from stable_baselines3 import TD3

from local_a3_agent_wrapper import SLICE_TYPES, quantize_a3_offset, normalize_slice_type


DEFAULT_STATE = {
    "gnb_id": 1,
    "neighbor_ids": [0, 2],
    "slice_types": ["eMBB", "URLLC", "mMTC"],
    "gnb_ids": [0, 1, 2],
    "bias": {
        "1->0:eMBB": -1.0,
        "1->2:eMBB": -1.0,
        "0->1:eMBB": 1.0,
        "2->1:eMBB": 1.0,
        "1->0:URLLC": 0.0,
        "1->2:URLLC": 0.0,
        "1->0:mMTC": 1.0,
        "1->2:mMTC": 1.0,
    },
    "counts": {
        "1:eMBB": 4,
        "0:eMBB": 1,
        "2:eMBB": 1,
        "1:URLLC": 2,
        "0:URLLC": 2,
        "2:URLLC": 2,
        "1:mMTC": 1,
        "0:mMTC": 4,
        "2:mMTC": 4,
    },
    "sla": {},
    "prev_offsets": {},
    "handover_failure_ratio": {},
    "pingpong_ratio": {},
    "loads": {
        "0:eMBB": 0.20,
        "1:eMBB": 0.90,
        "2:eMBB": 0.20,
        "0:URLLC": 0.45,
        "1:URLLC": 0.50,
        "2:URLLC": 0.45,
        "0:mMTC": 0.85,
        "1:mMTC": 0.20,
        "2:mMTC": 0.85,
    },
}


def _read_json(path: Path | None, inline: str | None) -> Dict[str, Any]:
    if path is None and not inline:
        return dict(DEFAULT_STATE)
    if path is not None and inline:
        raise ValueError("Use either --state-json or --state-inline, not both.")
    if path is not None:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return json.loads(str(inline))


def _key_variants(*parts: Any) -> Iterable[Any]:
    yield tuple(parts)
    yield ":".join(str(part) for part in parts)
    if len(parts) == 3:
        yield f"{parts[0]}->{parts[1]}:{parts[2]}"
    if len(parts) == 2:
        yield f"{parts[0]}:{parts[1]}"


def _lookup(mapping: Mapping[Any, Any] | None, *parts: Any, default: float = 0.0) -> float:
    data = mapping or {}
    for key in _key_variants(*parts):
        if key in data:
            return float(data[key])
    return float(default)


def _normalized_slice_types(raw: Sequence[str] | None) -> Tuple[str, ...]:
    values = raw or SLICE_TYPES
    return tuple(normalize_slice_type(item) for item in values)


def build_lower_observation(state: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build the LocalA3OffsetEnv observation from explicit state values.

    This mirrors LocalA3OffsetEnv._build_observation() without resetting or
    stepping the radio simulator.
    """
    gnb_id = int(state.get("gnb_id", 0))
    neighbor_ids = tuple(int(item) for item in state.get("neighbor_ids", []))
    slice_types = _normalized_slice_types(state.get("slice_types"))
    gnb_ids = tuple(int(item) for item in state.get("gnb_ids", [gnb_id, *neighbor_ids]))
    if not neighbor_ids:
        raise ValueError("state.neighbor_ids must contain at least one neighbor id.")
    if gnb_id not in gnb_ids:
        gnb_ids = (gnb_id, *gnb_ids)

    bias = state.get("bias", {})
    counts = state.get("counts", {})
    sla = state.get("sla", {})
    prev_offsets = state.get("prev_offsets", {})
    hf = state.get("handover_failure_ratio", {})
    pp = state.get("pingpong_ratio", {})
    loads = state.get("loads", {})
    k_ref = state.get("k_ref", {})

    obs = []
    blocks = []
    for neighbor_id in neighbor_ids:
        for slice_type in slice_types:
            ref = max(float(k_ref.get(slice_type, 20.0)), 1e-9)
            b_ij = float(np.clip(_lookup(bias, gnb_id, neighbor_id, slice_type), -1.0, 1.0))
            b_ji = float(np.clip(_lookup(bias, neighbor_id, gnb_id, slice_type), -1.0, 1.0))
            local_count = _lookup(counts, gnb_id, slice_type)
            neighbor_count = _lookup(counts, neighbor_id, slice_type)
            v_i = float(np.clip(_lookup(sla, gnb_id, slice_type), 0.0, 1.0))
            v_j = float(np.clip(_lookup(sla, neighbor_id, slice_type), 0.0, 1.0))
            prev = float(np.clip(_lookup(prev_offsets, neighbor_id, slice_type), -6.0, 6.0))
            hf_value = float(np.clip(_lookup(hf, neighbor_id, slice_type), 0.0, 1.0))
            pp_value = float(np.clip(_lookup(pp, neighbor_id, slice_type), 0.0, 1.0))

            obs.extend([
                b_ij,
                b_ji,
                local_count / ref,
                neighbor_count / ref,
                v_i,
                v_j,
                prev / 6.0,
                hf_value,
                pp_value,
            ])
            blocks.append({
                "neighbor_id": neighbor_id,
                "slice_type": slice_type,
                "bias": b_ij,
                "reverse_bias": b_ji,
                "local_count": local_count,
                "neighbor_count": neighbor_count,
                "previous_offset": prev,
            })

    for gid in gnb_ids:
        for slice_type in slice_types:
            obs.extend([
                float(_lookup(loads, gid, slice_type)),
                float(np.clip(_lookup(sla, gid, slice_type), 0.0, 1.0)),
            ])

    return np.asarray(obs, dtype=np.float32), {
        "gnb_id": gnb_id,
        "neighbor_ids": list(neighbor_ids),
        "slice_types": list(slice_types),
        "gnb_ids": list(gnb_ids),
        "blocks": blocks,
    }


def _bias_expectation(bias: float) -> str:
    if bias < -0.1:
        return "negative offset / encourage handover"
    if bias > 0.1:
        return "positive offset / retain traffic"
    return "near neutral"


def _is_bias_aligned(bias: float, applied_offset: float) -> bool:
    if abs(float(bias)) <= 0.1:
        return abs(float(applied_offset)) <= 2.0
    return float(bias) * (float(applied_offset) / 6.0) > 0.0


def _decode_action(action: np.ndarray, meta: Mapping[str, Any]) -> Dict[str, Any]:
    values = np.asarray(action, dtype=float).reshape(-1)
    neighbor_ids = list(meta["neighbor_ids"])
    slice_types = list(meta["slice_types"])
    block_by_key = {
        (int(block["neighbor_id"]), str(block["slice_type"])): block
        for block in meta.get("blocks", [])
    }
    rows = []
    idx = 0
    for neighbor_id in neighbor_ids:
        for slice_type in slice_types:
            raw = float(values[idx])
            applied = quantize_a3_offset(raw)
            block = block_by_key.get((int(neighbor_id), str(slice_type)), {})
            bias = float(block.get("bias", 0.0))
            rows.append({
                "neighbor_id": int(neighbor_id),
                "slice_type": str(slice_type),
                "upper_bias": bias,
                "raw_action_db": raw,
                "applied_a3_offset_db": applied,
                "meaning": (
                    "encourage handover"
                    if applied < 0.0
                    else "discourage handover"
                    if applied > 0.0
                    else "neutral"
                ),
                "expected_from_bias": _bias_expectation(bias),
                "bias_aligned": _is_bias_aligned(bias, applied),
            })
            idx += 1
    aligned = [bool(row["bias_aligned"]) for row in rows]
    return {
        "offsets": rows,
        "bias_alignment_rate": float(np.mean(aligned)) if aligned else 0.0,
    }


def _print_template() -> None:
    print(json.dumps(DEFAULT_STATE, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load a lower TD3 model and probe it with an explicit lower-agent "
            "state: upper biases, neighbor counts, SLA, previous offsets, and loads."
        )
    )
    parser.add_argument("--model", type=Path, help="Path to local_a3_td3_final.zip or shared_local_a3_td3_final.zip.")
    parser.add_argument("--state-json", type=Path, default=None, help="JSON file containing the probe state.")
    parser.add_argument("--state-inline", type=str, default=None, help="Inline JSON containing the probe state.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy prediction instead of deterministic.")
    parser.add_argument("--print-template", action="store_true", help="Print an editable example state JSON and exit.")
    args = parser.parse_args()

    if args.print_template:
        _print_template()
        return
    if args.model is None:
        raise SystemExit("--model is required unless --print-template is used.")

    state = _read_json(args.state_json, args.state_inline)
    obs, meta = build_lower_observation(state)
    model = TD3.load(str(args.model), device=args.device)

    expected_obs_shape = tuple(model.observation_space.shape)
    if obs.shape != expected_obs_shape:
        raise SystemExit(
            f"Observation shape mismatch: built {obs.shape}, model expects {expected_obs_shape}. "
            "Adjust neighbor_ids, slice_types, or gnb_ids in the state JSON."
        )

    action, _ = model.predict(obs, deterministic=not args.stochastic)
    decoded = _decode_action(action, meta)
    result = {
        "model": str(args.model),
        "observation_shape": list(obs.shape),
        "action_shape": list(np.asarray(action).shape),
        "controlled_gnb": int(meta["gnb_id"]),
        "input_blocks": meta["blocks"],
        **decoded,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
