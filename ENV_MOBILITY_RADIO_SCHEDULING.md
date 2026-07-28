# Mobility, Radio Parameters, and Scheduling in the 3-gNB HRL Simulator

This document explains how the simulation environment (`GlobalPPO3GNBEnv` →
`MultiGNBWrapper` → `NodeB`/`SliceL1*`/`SliceRAN*`) models **UE mobility**,
**radio-link/PRB parameters**, and the **nested time-stepping schedule** that
drives the whole loop. It is derived from reading the source directly; file
and line references are given throughout so claims can be re-verified as the
code evolves.

Core files:
- `global_ppo_3gnb_env.py` — outer PPO ("upper agent") environment, one action
  per **upper window**.
- `multi_gnb_wrapper.py` (`MultiGNBWrapper`) — the actual radio/mobility/A3
  simulator; everything below the upper agent runs here.
- `node_b.py` (`NodeB`) — one gNB: hexagonal coverage, path loss, per-slice L1.
- `channel_models.py` — trace-driven per-PRB SINR fading (`SINRSelectiveFading`)
  and the MCS/error-probability model (`MCSCodeset`).
- `schedulers.py` — `ProportionalFair`, the PRB scheduler.
- `slice_l1.py` / `slice_ran.py` — per-slice L1 multiplexing and RAN-level
  UE/queue/traffic/SLA bookkeeping.
- `safe_admission_controller.py` — per-window directional handover budget.
- `sumo_wrapper.py` — optional SUMO/TraCI mobility backend.
- `local_a3_agent_wrapper.py` — the "lower agent" interface that sets A3
  offsets and reads mobility outcomes.

---

## 1. The nested time hierarchy ("schedule")

There are three nested clocks, defined in `GlobalPPO3GNBEnv.__init__`
(`global_ppo_3gnb_env.py:120-251`):

```
upper_window_seconds                     one PPO action                default 1.0 s
  └─ local_steps_per_global               local (mobility+scheduler) steps  default 10
       local_step_seconds = upper_window_seconds / local_steps_per_global
       └─ radio_substeps                  radio ticks per local step   default 100
            radio_tick_seconds = local_step_seconds / radio_substeps
```

A consistency check enforces `radio_tick_seconds * radio_substeps ==
local_step_seconds` (`global_ppo_3gnb_env.py:239-251`). These become, inside
`MultiGNBWrapper`: `step_dt` (radio tick), `mobility_dt` (local step),
`radio_substeps` (`multi_gnb_wrapper.py:126-128`).

### 1.1 One `MultiGNBWrapper.step()` call = one local step

(`multi_gnb_wrapper.py:1151-1216`)

1. `_advance_gnbs()` — advances non-wrapper-managed L1 bookkeeping (mostly a
   no-op for the multi-gNB slices used here).
2. `_advance_mobility()` — moves every UE **once**, at `dt = mobility_dt`
   (§2).
3. `_evaluate_a3_handovers()` — evaluates the A3 event and executes at most
   `max_handovers_per_step` handovers (§4), immediately after mobility so
   handovers use fresh positions.
4. `_run_radio_substeps()` — runs `radio_substeps` radio ticks at
   `dt = step_dt` (§3.1). Traffic (packet arrivals) advances every tick;
   the full channel + PF-scheduler pass only runs every
   `_radio_sched_stride = min(5, radio_substeps)` ticks
   (`multi_gnb_wrapper.py:135`) — e.g. once every 5 of 100 ticks by default.
   Other ticks replay the cached allocation and only redo packet-level FIFO
   service. Substep 0 always triggers a full scheduler pass, so a handover
   from step 3 is guaranteed to be reflected immediately
   (`multi_gnb_wrapper.py:1283-1341`).
5. `_build_info()` assembles KPIs; `truncated` when `step_count >=
   max_episode_steps`.

### 1.2 One `GlobalPPO3GNBEnv.step()` call = one upper window

(`global_ppo_3gnb_env.py:665-830`)

1. The PPO action is decoded into a per-`(source, target, slice)` directional
   bias tensor and applied as A3 offsets. `begin_safe_admission_window(...)`
   freezes this window's directional handover quotas (§4.3); `begin_sla_window()`
   resets the SLA accounting window.
2. **Settle phase**: `effective_settle_steps` local steps run first
   (`base_env.step(0)` in a loop) so A3's time-to-trigger (TTT) can mature and
   handovers actually execute before any reward-relevant measurement begins.
   Offsets for directions whose quota is already exhausted are zeroed between
   local steps.
3. `begin_radio_measurement_window()` opens the PRB/load accounting window
   only **after** settling, then the remaining `measurement_steps` local
   steps run and their PRB/SLA/handover statistics feed the observation and
   reward.
4. Total local steps per window is `local_steps_per_global` (fixed) unless
   `dynamic_upper_window=True`, in which case it's sized from the accepted
   handover budget: `min(radio_meas_steps + required_exec_steps +
   post_handover_settle_steps, max_dynamic_local_steps_per_global)`
   (`global_ppo_3gnb_env.py:722-747`).

So, concretely, with defaults (`upper_window_seconds=1.0`,
`local_steps_per_global=10`, `radio_substeps=100`): **one PPO action = 10
local (100 ms) steps = 1000 radio ticks (1 ms each)**, with the PF scheduler
actually re-run roughly every 5 ticks (5 ms).

---

## 2. Mobility

### 2.1 Synthetic kinematics (default, `use_sumo_mobility=False`)

There is **no random-waypoint re-targeting loop** inside the wrapper. Motion
is plain constant-velocity straight-line kinematics:

- `_advance_mobility()` (`multi_gnb_wrapper.py:1241-1248`) calls, for every
  tracked UE, `ue.update_position(mobility_dt)` then `_apply_world_bounds(ue)`,
  then invalidates cached link metrics.
- `UE.update_position(dt)` (`slice_ran.py:277-279`):
  ```python
  self.x += self.vx * dt
  self.y += self.vy * dt
  ```
  `vx`/`vy` are **never resampled** by the wrapper — a UE keeps its velocity
  forever unless something external (a scenario, or SUMO sync) overwrites it.
- `_apply_world_bounds(ue)` (`multi_gnb_wrapper.py:721-734`) implements
  reflecting walls: if a UE exits `[x_min, x_max] × [y_min, y_max]`
  (computed once from the gNB layout's bounding box + 100 m margin,
  `multi_gnb_wrapper.py:112-121`), its position is clamped and the
  corresponding velocity component's sign is flipped so it heads back inward.

Initial `(x, y, vx, vy)` is entirely decided by the **caller** of `add_ue(...)`
(`multi_gnb_wrapper.py:1882-1932`, default `vx=vy=0`). In
`GlobalPPO3GNBEnv`, the curriculum/scenario path
(`_training_group_ue_state`, `global_ppo_3gnb_env.py:2879-2909`) places UEs
one of three ways:
- fixed offset from a source gNB, zero velocity (static),
- on a circle around a source gNB, zero velocity, when there's no handover
  target,
- along the straight line from a source gNB toward a target gNB at some
  `path_progress`, with lateral offset and per-UE fan-out, moving at
  `vx = speed_mps*ux, vy = speed_mps*uy` (unit vector source→target) — this
  is what drives UEs across cell boundaries to exercise handovers.

The snapshot/load-scenario path (`_initialize_load_scenario`,
`global_ppo_3gnb_env.py:2911-2928`) places UEs with `vx=vy=0` (static;
loads are set directly rather than via geometric movement).

A separate, **unrelated** random-position generator exists in
`slice_ran.py`/`channel_models.py` for the legacy single-cell
`SliceRANeMBB`/`SliceRANURLC` arrival flow: newly-arriving UEs get
`ue.x, ue.y = generate_xy(rng)` (rejection-sampled uniform point inside a
unit-normalized hexagon, `channel_models.py:73-87`) and Gaussian velocities
`ue.vx = rng.normal(0, 5)` (`slice_ran.py:459-461`, `481-483`). This path is
**bypassed** for the multi-gNB wrapper (`_build_slices_l1` marks mobile L1
slices `wrapper_managed=True`, and `SliceL1eMBB.slot()` early-returns in that
case — `scenario_creator.py:214-218`, `slice_l1.py:196-197`) — it only
applies to the legacy single-cell `create_env(..., multi_gnb=False)` path.

### 2.2 SUMO-based mobility (`use_sumo_mobility=True`)

`SumoMobilityWrapper` (`sumo_wrapper.py:12-209`) launches a SUMO subprocess
with `--step-length <mobility_dt>` and connects over TraCI on a free local
port; `.step()` calls `traci.simulationStep()` (advancing exactly one
`mobility_dt`) and returns per-entity `{x, y, speed, angle, road_id}` for all
vehicles and pedestrians — positions/velocities come from SUMO's own
road-network/car-following/pedestrian models, not from the RL-side code.

`MultiGNBWrapper._sync_sumo_mobility()` (`multi_gnb_wrapper.py:1685-1751`)
binds each SUMO vehicle/person to a wrapper `UE` (auto-creating one via
`add_ue` if unmapped), copies `x, y` directly, and derives `vx, vy` either
from `speed * sin(angle)/cos(angle)` or a position finite-difference if angle
is unavailable. Slice type for SUMO entities is assigned by hashing the
entity id into `[0,1)` (SHA-256-based, `multi_gnb_wrapper.py:678-694`) against
configurable vehicle/person slice mixes. Departed SUMO entities are detached;
`_ensure_minimum_sumo_ues` anchors placeholder UEs at gNB positions if the
live population drops too low.

### 2.3 UE-to-gNB association

- **Coverage test**: `NodeB.is_point_in_coverage(x, y)`
  (`node_b.py:110-132`) — distance check against `coverage_radius`, plus a
  hexagon point-in-polygon ray-cast over the six vertices computed in
  `_calculate_hexagon_vertices` (`node_b.py:88-101`).
- **Initial/attach-time serving cell**: `_find_best_gnb_for_ue(ue)`
  (`multi_gnb_wrapper.py:2541-2549`) picks the gNB with the highest
  instantaneous SINR (best-SINR association, not RSRP) among in-coverage
  cells; used at `reset()` and `add_ue()`.
- **Ongoing mobility changes serving cell only through A3 handovers** (§4),
  not automatically when a UE leaves coverage.

---

## 3. Radio parameter calculation pipeline

### 3.1 Per-radio-tick flow

`_run_radio_substeps()` (`multi_gnb_wrapper.py:1283-1341`), for each of
`radio_substeps` ticks:

1. `_advance_traffic_one_substep()` calls `ue.traffic_step()` for every UE
   (`slice_ran.py:172-211`): generates new packets from the UE's
   `traffic_source` (CBR/VBR, see `traffic_generators.py`), timestamps and
   FIFO-enqueues them, drops overflow past `buffer_size`, updates
   head-of-line delay (`hol_delay_s`).
2. **On scheduled ticks only** (`i % _radio_sched_stride == 0`, stride =
   `min(5, radio_substeps)`): run the full channel+scheduler pass
   `_simulate_radio_and_service()` (§3.2) and cache each UE's realized
   `scheduled_bits`, `rx_probability`, `allocated_prbs`.
3. **On replay ticks**: restore the cached allocation, draw
   `received = rng.random() < cached_rx_prob`, and call
   `ue.transmission_step(received)` directly — i.e. the expensive
   channel/scheduler computation is amortized across a few ms, but
   packet-level FIFO service still advances every 1 ms tick.
4. Delay/SLA bookkeeping is updated every tick: `_update_delay_after_service()`,
   `_accumulate_radio_measurement_sample()` (feeds the upper-window average
   load), `ue.update_sla_expirations()` (per-packet deadline misses: 10 ms
   for URLLC, 1 s for mMTC — `slice_ran.py:247-253`), and
   `_accumulate_sla_window()`.

### 3.2 Link-metric computation (`_compute_link_metrics`, `multi_gnb_wrapper.py:2649-2723`)

For a UE served/considered by gNB `g`:

```
env_loss_db  = _environment_loss_db(ue)                     # optional degradation zones
rx_total_dbm = gnb.get_received_power_dbm(ue.x, ue.y) - env_loss_db
rx_dbm       = rx_total_dbm_per_prb - env_loss_db            # rx_total_dbm - 10*log10(n_prbs)
noise_dbm    = gnb.get_noise_power_dbm()                     # -174 + 10log10(BW_per_RB) + NF
sig_w, noise_w = dbm_to_watts(rx_dbm), dbm_to_watts(noise_dbm)
interf_w     = _compute_interference_watts(gnb, ue)           # same-carrier neighbors
snr_db       = rx_dbm - noise_dbm
sinr_db      = clip(10*log10(sig_w / max(noise_w + interf_w, 1e-15)), -20, 40)
rsrp_dbm     = rx_total_dbm                                   # wideband, used for A3
```

`NodeB.get_received_power_dbm(x, y)` (`node_b.py:187-215`) uses an
urban-macro path-loss model:
```
path_loss_db = 128.1 + 37.6*log10(d_km) + 20*log10(f_ghz / 2.0)   # mild frequency correction
path_loss_db = max(path_loss_db + shadowing_db, 70.0)              # min coupling loss clamp
rx_dbm = tx_power_dbm - path_loss_db
```
(`shadowing_std_db` defaults to 0 — the dense scenario uses trace-driven
fading instead, see below.) `NodeB.get_noise_power_dbm()` (`node_b.py:178-185`)
is thermal noise: `N = -174 dBm/Hz + 10log10(BW) + NF`.

**Interference** (`_compute_interference_watts`,
`multi_gnb_wrapper.py:2628-2647`): only gNBs sharing the same `carrier_id`
(`NodeB.uses_same_carrier`, `node_b.py:226-230`) are interferers. For each
same-carrier neighbor in coverage of the UE, its per-PRB received power at
the UE is scaled by that neighbor's **PRB duty-cycle activity** — the
fraction of its PRBs actually allocated in the *previous* scheduler pass
(`_gnb_allocated_prb_activity`, `multi_gnb_wrapper.py:2375-2379`) — and summed
across interferers. This is a duty-cycle-weighted interference estimate, not
a static frequency-reuse-1 worst case.

### 3.3 Frequency-selective fading and MCS (`channel_models.py`)

`SINRSelectiveFading` (`channel_models.py:142-206`) provides per-PRB fading
offsets sampled from one of three CSV traces
(`fading_trace_{EPA_3kmph,ETU_3kmph,EVA_60kmph}.csv`) — each user is assigned
a random trace type, start index, and step direction at insertion; each call
to `get_snr(user_id)` steps through the trace (wrapping/rerandomizing at
boundaries) and returns `fading_vector + nominal_sinr` (an `n_prbs`-length
vector). `NominalSINR` (`channel_models.py:122-140`) supplies the scalar
nominal-SINR baseline via a TS 36.942 macro-cell/free-space model driven by a
synthetic hexagon-position sample (this legacy nominal-SINR generator is
separate from `NodeB`'s path-loss formula and is mainly used by the
single-cell/legacy path and as the additive baseline for the fading traces).

In the multi-gNB wrapper, `_frequency_selective_snr_vector(ue, nominal_sinr_db,
n_prbs)` (`multi_gnb_wrapper.py:2135-2161`) adds these per-PRB trace offsets
to the scalar `sinr_db` computed in §3.2, producing `ue.snr` — the vector fed
to the scheduler.

`MCSCodeset` (`channel_models.py:268-336`) loads an MCS table (rate, order,
modulation, reference SNR) from CSV and models packet reception probability
as a sigmoid around each MCS's reference SNR:
`mcs_rate_vs_error(snr, error_bound)` returns the highest MCS whose estimated
error is below the bound; `effective_snr(mcs, snr_vector)` averages
per-PRB SNRs in the *mutual-information* domain (via per-modulation sigmoid/
inv-sigmoid parameters) rather than linearly, when multiple PRBs are
assigned.

### 3.4 PF scheduler and useful/wasted/demand PRBs

**MCS-scheduler path** (default, `_simulate_radio_and_service`,
`multi_gnb_wrapper.py:2206-2373`), per serving gNB, per scheduled tick:

1. UEs with non-finite or too-low SINR (`<= disconnect_sinr_db`) are
   force-disconnected (`prbs=useful_prbs=wasted_prbs=bits=0`).
2. Each remaining UE gets its frequency-selective SNR vector (§3.3).
3. `ProportionalFair.allocate(ues, n_prb)` (`schedulers.py:51-186`) runs once
   for the whole gNB:
   - Wideband MCS pre-selection from `ue.e_snr` gives an achievable rate
     `ue_rate = sym_per_prb(158) * bits_per_symbol`.
   - PRBs are handed out `granularity=2` at a time to
     `argmax(ue_rate * (queue>0) / ue_th)` — the classic PF metric, achievable
     rate over an exponentially-smoothed average throughput `ue_th`
     (`ue_th = a*ue_th + b*bits/slot_length`, `b = 1/window_ticks`, window
     set by `pf_averaging_window_s`). Allocation stops once the best score is
     ≤ 0 (no UE has both data and rate).
   - **Post-allocation MCS re-selection**: using the SNR samples of the
     *actually assigned* PRBs (not the wideband estimate), the scheduler
     scans MCS indices upward and keeps the highest one whose estimated
     reception probability still meets the target; `realized_bits =
     min(prbs * realized_rate, queue)`.
   - `useful_prbs = min(prbs, ceil(realized_bits / realized_rate))` if bits
     were served, else 0; `wasted_prbs = prbs - useful_prbs` — i.e. PRBs
     granted but not needed because the queue drained before using the full
     grant.
4. Reception is a Bernoulli draw against the realized MCS's error
   probability; `ue.transmission_step(received)` drains the FIFO queue.
5. The realized PRB activity feeds next-tick's interference duty factor
   (§3.2).

**Shannon fallback path** (used only if `channel_models`/`schedulers` fail to
import, or `use_mcs_scheduler=False`): greedily allocates
`ceil(queue_bits / bits_per_prb)` PRBs per UE (sorted by queue then SINR)
using `bits_per_prb = 180e3 * step_dt * clip(log2(1+sinr_linear), 0, 8)`.

**Demand PRBs** (a third, independent figure from allocated/useful) are
computed by the wrapper directly from queue backlog and the slice-specific
MCS codeset (`_estimate_queue_demand_prbs`, `multi_gnb_wrapper.py:427-432`):
`demand_prbs = ceil(queue_bits / bits_per_prb_estimate)`. This is what the
upper agent's "load" observations and reward terms are built from
(`slice_kpis[...]["demand_load"]`), since it reflects offered traffic
regardless of whether the cell currently has capacity to serve it.

### 3.5 L1/RAN slice bookkeeping (`slice_l1.py`, `slice_ran.py`)

- **`SliceL1eMBB`/`SliceL1URLLC`** multiplex one or more `SliceRAN*` slices
  over a shared `n_prbs` budget. When `wrapper_managed=True` (the multi-gNB
  case), `slot()` is a no-op — all traffic/radio/scheduling happens in
  `MultiGNBWrapper` instead; this class matters for the legacy single-cell
  path, where it calls `ProportionalFair.allocate` itself.
- **`SliceL1mMTC`** models aggregate NB-IoT-style devices directly (repeated
  transmissions over shared carriers, one PRB per active device), separate
  from the mobile UE/queue model.
- **`UE`** (`slice_ran.py:38-292`) owns a `deque[Packet]` FIFO queue,
  timestamped packet arrival/service/drop accounting, per-slice SLA deadlines
  (URLLC 10 ms, mMTC 1 s, eMBB none), and radio-state fields (`snr`, `e_snr`,
  `sinr`, `prbs`, `mcs`, `p`).
- **SLA severity** in the multi-gNB path is aggregated per `(gNB, slice)`
  window (`_sla_window_metrics_for_key`, `multi_gnb_wrapper.py:524-566`):
  for eMBB, `severity = max(0, threshold - delivered/offered)/threshold`
  (threshold default 0.80); for URLLC/mMTC, `severity = clip(failed/generated,
  0, 1)` against a max-failure-ratio threshold (default 0.01 for both).

---

## 4. Handover: A3 event, TTT, and admission control

### 4.1 A3 event and time-to-trigger

`_evaluate_a3_handovers()` (`multi_gnb_wrapper.py:1375-1632`) runs once per
local step, right after mobility. For each connected UE and each in-coverage
neighbor gNB:

```
threshold = rsrp_serving + a3_offset(slice) + a3_hysteresis_db
if rsrp_neighbor > threshold:
    ttt_count = serving_gnb.tick_a3_counter(ue_id, neighbor_id)   # NodeB, node_b.py:78-81
    eligible = ttt_count >= handover_ttt                          # default 3 consecutive local steps
else:
    serving_gnb.reset_a3_counter(ue_id, neighbor_id)              # node_b.py:75-76
```

`a3_offset` is the slice-aware value set by the upper/lower agent via
`set_a3_offset(source, target, slice)`; **negative offset makes handover
easier** (lowers the RSRP bar), positive makes it harder. RSRP for both cells
comes from `_compute_link_metrics(...)["rsrp_dbm"]` (§3.2), i.e. current UE
position and the current path-loss/shadowing state.

### 4.2 Stability guards (cooldown / residence / ping-pong)

All A3-related second-based parameters are converted to **wrapper-tick**
(local-step) counts at construction, using `a3_tick_s = mobility_dt`:

- **Cooldown** (`a3_handover_cooldown_s`): a UE that just handed over is
  fully excluded from A3 evaluation for this many local steps.
- **Minimum residence** (`a3_min_residence_s`, ≥ cooldown): after cooldown
  expires, still excluded unless serving SINR has dropped to/below
  `a3_emergency_sinr_db` (emergency override).
- **Ping-pong guard** (`a3_pingpong_guard_s`, ≥ min residence): actively
  *blocks* (not just logs) a candidate handover that would return a UE
  directly to its immediately-previous serving cell within this window,
  unless serving SINR is already poor.
- **Ping-pong logging** (`a3_pingpong_threshold_s`): a completed handover is
  separately logged as a ping-pong (for reward/diagnostics) if the UE
  returns to its prior cell within this many local steps.
- **History window** (`a3_history_window_s`): sliding window used to compute
  recent handover-failure and ping-pong *ratios*, which feed candidate
  ranking.

Eligible candidates additionally carry an `a3_margin = rsrp_neighbor -
rsrp_serving - offset - hysteresis` and are checked against episode-level
budgets (`max_handovers_per_episode`, `max_handovers_per_ue_episode`).

### 4.3 Directional admission budget (`safe_admission_controller.py`)

`DirectionalAdmissionBudgetController` (aliased `SafeAdmissionController`) is
explicitly **not** a traffic-safety veto — per its own docstring, A3 already
determines *radio* eligibility; this controller only limits *how many*
eligible handovers may execute per `(source, target, slice)` direction within
one upper window:

- `begin_upper_window(...)` freezes quotas from the directional bias tensor:
  `strength = max(0, -bias)`; if `strength <= bias_deadband` (default 0.05)
  the direction gets **zero** quota, otherwise
  `direction_budget = min(ceil(strength * source_ue_count), source_ue_count)`
  — i.e. quota is a bias-proportional share of the source cell's UEs of that
  slice.
- `admit_candidates(...)` ranks eligible A3 candidates by
  `(a3_margin, -pingpong_ratio, -handover_failure_ratio)` and admits up to
  the per-step and per-direction remaining quota; **quota is only
  provisionally reserved here**.
- `commit(candidate)` is the only place that permanently consumes quota —
  called only after `_perform_handover` confirms the handover actually
  succeeded. If a direction's quota becomes exhausted mid-window, its A3
  offset is immediately neutralized so no further TTT accumulates there for
  the rest of the window.
- If `safe_admission_enabled=False`, this whole layer is bypassed and A3
  candidates are simply rank-sorted by `a3_margin` and truncated to
  `max_handovers_per_step`.

### 4.4 Executing a handover

`_perform_handover` (`multi_gnb_wrapper.py:1634-1651`, mirrored in the
lower-agent's own `LocalA3OffsetEnv._handover_ue`,
`local_a3_agent_wrapper.py:324-379`): detach from the old `NodeB`, attach to
the new one via `NodeB.attach_ue` (routes into the matching L1 slice by
`slice_type`, with mMTC UEs attached to an eMBB/URLLC L1 for mobility
purposes while keeping `slice_type == mMTC` for load accounting,
`node_b.py:532-548`). On success, both gNBs' A3 TTT counters for that UE are
cleared, per-UE/episode handover counters increment, and ping-pong is logged
if applicable.

---

## 5. Upper agent ↔ lower agent responsibilities

- **Upper agent (`GlobalPPO3GNBEnv`, PPO)**: once per upper window, outputs a
  continuous directional bias tensor `B[source, neighbor_slot, slice] ∈
  [-1, 1]`. This bias (a) is converted into A3 offsets (via
  `strong_directional_heuristic_local_executor` /
  `_compute_strong_local_offsets`) that make handover easier/harder per
  direction, and (b) sets the directional admission quotas (§4.3). Reward is
  built from PRB-load balance, excess-load, SLA severity, SINR-deficit, and
  handover/ping-pong penalties measured over the post-settle measurement
  window (`global_ppo_3gnb_env.py` reward block, not detailed here).
- **Lower/local agent (`LocalA3OffsetEnv`)**: given the upper bias as context,
  outputs per-neighbor-slice continuous proto-offsets in `[-6, 6]` dB,
  quantized to `{-6,-4,-2,0,2,4,6}` dB (`quantize_a3_offset`,
  `local_a3_agent_wrapper.py:27-32`), which become the actual A3 offsets fed
  into `_evaluate_a3_handovers`. It also directly executes handovers via a
  simplified A3 check when trained standalone
  (`_execute_a3_handovers`, `local_a3_agent_wrapper.py:261-322`).

---

## 6. Key defaults reference

| Parameter | Default | Meaning |
|---|---|---|
| `upper_window_seconds` | 1.0 s | One PPO action's duration |
| `local_steps_per_global` | 10 | Local (mobility+A3) steps per upper window |
| `radio_substeps` | 100 | Radio ticks per local step |
| `_radio_sched_stride` | `min(5, radio_substeps)` | Ticks between full PF-scheduler runs |
| `handover_ttt` | 3 | Consecutive local steps the A3 condition must hold |
| `a3_handover_cooldown_s` | 2.0–5.0 (caller-dependent) | Post-handover full exclusion |
| `a3_min_residence_s` | 2.0–15.0 | Post-handover exclusion unless emergency SINR |
| `a3_pingpong_guard_s` | ≥ min residence | Blocks direct-return handovers |
| `disconnect_sinr_db` | wrapper default | SINR floor below which a UE is force-disconnected |
| `pf_averaging_window_s` | 0.25 s | PF throughput-smoothing window |
| `coverage_radius` | 500–520 m | Hexagonal cell radius |
| `center_frequency_hz` / `bandwidth_hz` | 3.5 GHz / 20 MHz | Carrier parameters (`NodeB`) |
| `tx_power_dbm` / `noise_figure_db` | 30 dBm / 7 dB | Link budget inputs |

---

## 7. Practical notes / gotchas

- **"Load" has three distinct meanings** in this codebase: *allocated* PRBs
  (what the scheduler granted), *useful* PRBs (allocated minus wasted, i.e.
  what was actually needed to drain the queue), and *demand* PRBs (queue
  backlog converted to PRBs regardless of what was granted). The upper
  agent's reward and observations are demand-based; scheduler/radio
  evaluation metrics are useful/allocated-based.
- **The channel/scheduler pass is not run every radio tick** — it's
  amortized over `_radio_sched_stride` ticks (5 by default) for performance;
  only packet FIFO service and SLA-deadline bookkeeping run every tick.
- **A3 offsets act on RSRP, not on PRB load directly** — the upper agent's
  bias only becomes a load-balancing effect indirectly, by making cell edges
  favor one neighbor over another; the admission controller's directional
  quota is what actually caps *how many* UEs can be moved per window.
- **Mobility is not randomly perturbed at each step** in the non-SUMO path —
  velocity is set once at spawn (or by a scenario's path/`speed_mps`) and
  UEs move in straight lines (reflecting off world bounds) until an A3
  handover changes their serving cell or the episode ends.
