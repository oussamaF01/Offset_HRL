# Project Summary — 5G Load Balancing with A3 Offset Heuristic

---

## 1. Core Problem

The simulation always showed **gNB-0 eMBB load = 1.0** regardless of how many UEs were present or what bit rate they used.

### Root Cause — `step_dt` mismatch

The scheduler runs **once per step**, serving `n_prbs × bits_per_prb` bits per step.  
With `step_dt = 0.1 s` (old), each UE accumulates 100 ms of traffic demand but the scheduler only serves 1 ms of capacity:

| `step_dt` | Cell capacity per step | 1 UE @ 12 Mbps demand | Result |
|---|---|---|---|
| `0.1 s` | `100 × 853 / 0.1 ≈ 0.85 Mbps` | `1.2 Mbps > 0.85` | **load = 1.0 always** |
| `0.001 s` | `100 × 853 / 0.001 ≈ 85 Mbps` | `0.12 Mbps << 85` | **load = 0.15** ✓ |

**Fix:** `TICK_S = 0.001` everywhere (1 ms per tick matches `slot_length`).

---

## 2. Load Calculation

```
load = useful_prbs / n_prbs
```

- `useful_prbs = min(allocated_prbs, ceil(queue / bits_per_prb))` — actual data PRBs, not just allocated
- `bits_per_prb ≈ 853` bits at high SINR (MCS lookup)
- `n_prbs = 100` (default per gNB)

### Deterministic load rule

Set `packet_size_bits = bit_rate × step_dt` → exactly 1 packet per tick → **exactly N×15 PRBs** for N eMBB UEs at 12 Mbps:

| UEs at gNB | PRBs used | Load |
|---|---|---|
| 1 | 15 | 0.15 |
| 2 | 30 | 0.30 |
| 3 | 45 | 0.45 |
| 4 | 60 | 0.60 |
| 5 | 75 | 0.75 |

---

## 3. Inter-Cell Interference Problem

With all gNBs on the **same `carrier_id = 0`**, each UE's SINR is degraded by interference from neighbouring gNBs:

```
SINR = signal_from_serving / (interference_from_others + noise)
```

A UE at 150 m from gNB-0 (other gNBs 300–400 m away) sees significant interference → lower `bits_per_prb` → needs **more PRBs** for the same bit rate → load > expected.

### Fix — separate carrier IDs

```python
# Each gNB on its own carrier → no inter-cell interference → clean SINR
GNB_CONFIGS = [
    {'id': 0, ..., 'carrier_id': 0},
    {'id': 1, ..., 'carrier_id': 1},
    {'id': 2, ..., 'carrier_id': 2},
]
```

With separate carriers: load = exactly `N × 0.15` regardless of UE position within coverage.

---

## 4. A3 Handover Mechanism

The A3 condition triggers a handover when (sustained for TTT):

```
RSRP(target_gNB) > RSRP(serving_gNB) + A3_offset + hysteresis
```

| A3 offset | Effect | HO boundary |
|---|---|---|
| −6 dB | target looks stronger → **HO earlier** | ~195 m (before midpoint) |
| 0 dB | natural boundary | ~225 m (midpoint) |
| +6 dB | target must be stronger → **HO later** | ~276 m (past midpoint) |

Each 6 dB shifts the HO boundary by ~40 m.

---

## 5. Over-Migration Bug & Fix

### The problem

With gNB-0 overloaded (load = 0.75) and gNB-1 empty (load = 0):

1. Heuristic computes maximum imbalance → `bias = −1.0`
2. `raw_offset = bias × 12.0 = −12 dB` — fires **all 5 UEs' A3 conditions simultaneously**
3. All 5 TTT counters expire within 0.5 s → all 5 HOs → gNB-1 saturates
4. Hard veto (line 443) never triggers because gNB-1 stays below `l_safe = 0.80` until the 5th UE arrives

### Fix in `strong_heuristic_local_executor.py`

**Change 1 — gentler negative scale (line 420):**
```python
# Before
raw_offset = bias * 12.0   # −12 dB max → all UEs cross boundary at once

# After
raw_offset = bias * 4.0    # −4 dB max → UEs stagger HOs one at a time
```

**Change 2 — target headroom veto (line 443):**
```python
# Before
if proto_offset < 0.0 and (not target_is_safe or not radio_feasible):
    proto_offset = 0.0

# After — also veto when target can't absorb one more UE
target_has_headroom = (target_load + 0.15) < l_safe
if proto_offset < 0.0 and (not target_is_safe or not target_has_headroom or not radio_feasible):
    proto_offset = 0.0
```

### Result after fix (with separate carrier IDs)

```
t= 1s  n0=5 n1=0  g0=0.75  g1=0.00  offset=−4 dB  ← imbalanced, push
t= 9s  n0=4 n1=1  g0=0.60  g1=0.15  offset=−4 dB  ← still imbalanced
t=11s  n0=3 n1=2  g0=0.45  g1=0.30  offset=−3 dB  ← gap closing
t=14s  n0=2 n1=3  g0=0.30  g1=0.45  offset= 0 dB  ← balanced, STOP
t=20s  n0=1 n1=4  g0=0.15  g1=0.60  offset=+2 dB  ← resist further HOs
```

No cell ever exceeds 0.75 load. HOs are staggered, not simultaneous.

---

## 6. Notebooks Created This Session

| Notebook | Purpose |
|---|---|
| `three_gnb_mixed_slice_load.ipynb` | Validate load formula: 1 eMBB + 2 URLLC at gNB-0, load = 0.15 + 0.12 |
| `ue_transfer_load_shift.ipynb` | Show load change as UEs are manually transferred gNB-0 → gNB-1 in 0.15 steps |
| `a3_offset_handover_demo.ipynb` | Compare offset = −6 / 0 / +6 dB: shows HO at x=195 / 236 / 276 m |

---

## 7. Key Parameters (correct values)

```python
TICK_S            = 0.001        # must match slot_length (1 ms)
packet_size_bits  = bit_rate * TICK_S   # exactly 1 packet/tick → deterministic load
carrier_id        = unique per gNB      # avoid inter-cell interference

# In strong_heuristic_local_executor.py
raw_offset = bias * 4.0    # negative scale (was 12.0)
raw_offset = bias * 6.0    # positive scale (unchanged)
target_headroom_check = (target_load + 0.15) < l_safe   # new veto condition
```

---

## 8. Useful Code Snippets

### Check load at any tick
```python
loads = env.get_slice_loads()   # dict[(gnb_id, slice_type)] → float
# e.g. loads[(0, 'eMBB')] = 0.75
```

### Force a handover (demo/test use)
```python
ue.x, ue.y = 430.0, 10.0          # reposition near target gNB
env._perform_handover(ue, env._gnb_by_id[from_id], env._gnb_by_id[to_id])
ue.queue = 0.0                     # clear backlog
```

### Set A3 offset
```python
env.set_a3_offset(serving_gnb_id, target_gnb_id, slice_type, offset_db)
# e.g. env.set_a3_offset(0, 1, 'eMBB', -4.0)  → encourage HO gNB-0→gNB-1
```
