"""
Verify that the scheduler allocates exactly as many useful PRBs as the UE
demands — no more, no less — and that allocation tracks demand when the
backlog changes.

Fixed scenario:
  - 1 gNB, 1 static UE (vx=vy=0), radio_substeps=1
  - deterministic CBR traffic → predictable queue growth per step
  - UE placed 10 m from gNB → stable high-MCS → constant bits_per_prb

Within every substep the order is:
  1. traffic_step()  →  queue grows by new_bits
  2. scheduler       →  queue drained by served_bits, useful_prbs set
"""

import numpy as np
import pytest
from scenario_creator import create_nodeb
from multi_gnb_wrapper import MultiGNBWrapper
from channel_models import MCSCodeset


SYM_PER_PRB = 158   # default in ProportionalFair


def _make_env(bit_rate: float):
    rng = np.random.default_rng(42)
    gnb = create_nodeb(
        rng, 0,
        slots_per_step=1,
        slot_length=1e-3,
        L1_level=False,
        node_id=0,
        node_x=0.0, node_y=0.0,
        coverage_radius=900,
        n_prbs_override=100,
        wrapper_managed_mobile_slices=True,
    )
    env = MultiGNBWrapper(
        [gnb],
        step_dt=1e-3,
        mobility_dt=1e-3,
        radio_substeps=1,           # one scheduler tick per env.step()
        disconnect_sinr_db=-100.0,  # never drop the UE
        ue_traffic_profiles={
            "eMBB": {
                "traffic_model": "fixed_packet_cbr",
                "packet_size_bits": 12_000.0,
                "bit_rate": float(bit_rate),
            }
        },
    )
    ue_id = env.add_ue(x=10.0, y=0.0, vx=0.0, vy=0.0, slice_type="eMBB")
    ue = env.get_ue(ue_id)

    # manually attach to gNB0 and mark connected
    for g in env.gnbs:
        g.detach_ue(ue_id)
    gnb.attach_ue(ue)
    ue.serving_gnb = 0
    ue.connected = True
    return env, ue_id


def _bits_per_prb(ue) -> float:
    codeset = MCSCodeset()
    _, bits_per_sym = codeset.mcs_rate_vs_error(float(ue.e_snr), 0.1)
    return SYM_PER_PRB * bits_per_sym


# ---------------------------------------------------------------------------
# Standalone runner — print a trace table
# ---------------------------------------------------------------------------
def run_trace(bit_rate: float = 5_000_000.0, steps: int = 30):
    env, ue_id = _make_env(bit_rate)
    ue = env.get_ue(ue_id)

    print(f"\n{'step':>4} | {'queue_before':>12} | {'new_bits':>10} | "
          f"{'queue_after':>11} | {'served':>10} | "
          f"{'useful_prbs':>11} | {'expected_prbs':>13} | match")
    print("-" * 95)

    for i in range(steps):
        queue_before_step = float(ue.queue)

        env.step(0)
        m = env.get_ue_radio_metrics(ue_id)

        # queue before scheduler = queue after step + what the scheduler drained
        # (traffic_step fires first inside the substep, then the scheduler)
        queue_before_sched = float(m["queue"]) + float(m["served_bits"])
        new_bits = queue_before_sched - queue_before_step

        bpp = _bits_per_prb(ue)
        expected_prbs = int(np.ceil(queue_before_sched / bpp)) if queue_before_sched > 0 else 0

        match = "OK" if m["useful_prbs"] == expected_prbs else f"MISMATCH (got {m['useful_prbs']})"

        print(
            f"{i:>4} | {queue_before_step:>12.0f} | {new_bits:>10.0f} | "
            f"{m['queue']:>11.0f} | {m['served_bits']:>10.0f} | "
            f"{m['useful_prbs']:>11d} | {expected_prbs:>13d} | {match}"
        )

    env.close()


# ---------------------------------------------------------------------------
# Pytest assertions
# ---------------------------------------------------------------------------
def test_useful_prbs_matches_demand_exactly():
    """useful_prbs must equal ceil(queue_before_scheduler / bits_per_prb)."""
    env, ue_id = _make_env(bit_rate=5_000_000.0)
    ue = env.get_ue(ue_id)

    mismatches = []
    for i in range(50):
        queue_before_step = float(ue.queue)
        env.step(0)
        m = env.get_ue_radio_metrics(ue_id)

        queue_before_sched = float(m["queue"]) + float(m["served_bits"])
        bpp = _bits_per_prb(ue)
        expected = int(np.ceil(queue_before_sched / bpp)) if queue_before_sched > 0 else 0

        if m["useful_prbs"] != expected:
            mismatches.append((i, expected, m["useful_prbs"]))

    env.close()
    assert not mismatches, f"PRB mismatches at steps: {mismatches}"


def test_useful_prbs_never_exceeds_allocated():
    """Scheduler must never report more useful PRBs than it allocated."""
    env, ue_id = _make_env(bit_rate=5_000_000.0)
    ue = env.get_ue(ue_id)

    for _ in range(50):
        env.step(0)
        m = env.get_ue_radio_metrics(ue_id)
        assert m["useful_prbs"] <= m["allocated_prbs"], (
            f"useful_prbs={m['useful_prbs']} > allocated_prbs={m['allocated_prbs']}"
        )

    env.close()


def test_zero_queue_means_zero_prbs():
    """UE with empty queue must not be allocated any PRBs."""
    env, ue_id = _make_env(bit_rate=0.0)   # no traffic
    ue = env.get_ue(ue_id)
    ue.queue = 0.0

    for _ in range(10):
        env.step(0)
        m = env.get_ue_radio_metrics(ue_id)
        assert m["useful_prbs"] == 0, f"Expected 0 useful PRBs, got {m['useful_prbs']}"

    env.close()


def test_prbs_increase_when_backlog_spikes():
    """Injecting a large backlog must cause useful_prbs to increase immediately."""
    env, ue_id = _make_env(bit_rate=1_000_000.0)
    ue = env.get_ue(ue_id)

    # warm up — let queue reach steady state
    for _ in range(10):
        env.step(0)
    m_before = env.get_ue_radio_metrics(ue_id)
    prbs_before = m_before["useful_prbs"]

    # inject a large backlog spike
    ue.queue += 500_000.0

    env.step(0)
    m_after = env.get_ue_radio_metrics(ue_id)

    assert m_after["useful_prbs"] > prbs_before, (
        f"PRBs should have increased after backlog spike: "
        f"{prbs_before} -> {m_after['useful_prbs']}"
    )

    env.close()


if __name__ == "__main__":
    run_trace(bit_rate=5_000_000.0, steps=30)
