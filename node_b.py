#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz
Enhanced with hexagonal coverage + radio parameters for multi-gNB simulation
"""

import numpy as np
from typing import Dict, Tuple


class NodeB:
    def __init__(
        self,
        id,
        x,
        y,
        slices_l1,
        slots_per_step,
        n_prbs,
        coverage_radius=500,
        slot_length=1e-3,
        carrier_id=0,
        center_frequency_hz=3.5e9,
        bandwidth_hz=20e6,
        tx_power_dbm=30.0,
        noise_figure_db=7.0,
        shadowing_std_db=0.0,
    ):
        """
        Initialize gNodeB with hexagonal coverage and radio parameters.

        Args:
            id: gNodeB identifier
            x, y: center coordinates of the hexagon
            slices_l1: list of L1 slice objects
            slots_per_step: number of time slots per step
            n_prbs: total number of Physical Resource Blocks
            coverage_radius: radius of the circumscribed circle (distance from center to vertices)
            slot_length: duration of each time slot
            carrier_id: identifier of carrier/band used by this gNodeB
            center_frequency_hz: center frequency in Hz
            bandwidth_hz: bandwidth in Hz
            tx_power_dbm: transmit power in dBm
            noise_figure_db: receiver noise figure in dB
            shadowing_std_db: standard deviation for extra random log-normal
                shadowing. Keep 0 when using trace-driven fading/shadowing.
        """
        self.id = id
        self.x = x
        self.y = y
        self.coverage_radius = coverage_radius
        self.slices_l1 = slices_l1
        self.n_slices_l1 = len(self.slices_l1)
        self.slots_per_step = slots_per_step
        self.n_prbs = n_prbs
        self.slot_length = slot_length

        # Radio parameters
        self.carrier_id = carrier_id
        self.center_frequency_hz = center_frequency_hz
        self.bandwidth_hz = bandwidth_hz
        self.tx_power_dbm = tx_power_dbm
        self.noise_figure_db = noise_figure_db
        self.shadowing_std_db = float(shadowing_std_db)

        # Pre-calculate hexagon vertices for visualization
        self.vertices = self._calculate_hexagon_vertices()

        # Calculate coverage area
        self.coverage_area = self._calculate_hexagon_area()
        self._a3_counters: Dict[Tuple[int, int], int] = {}
        self.reset()

    def reset_a3_counter(self, ue_id: int, neighbor_id: int) -> None:
        self._a3_counters.pop((int(ue_id), int(neighbor_id)), None)

    def tick_a3_counter(self, ue_id: int, neighbor_id: int) -> int:
        key = (int(ue_id), int(neighbor_id))
        self._a3_counters[key] = self._a3_counters.get(key, 0) + 1
        return self._a3_counters[key]

    def clear_all_a3_counters_for_ue(self, ue_id: int) -> None:
        keys_to_remove = [key for key in self._a3_counters if key[0] == int(ue_id)]
        for key in keys_to_remove:
            self._a3_counters.pop(key, None)

    def _calculate_hexagon_vertices(self):
        """
        Calculate the six vertices of the regular hexagon.
        Returns:
            list of (x, y) tuples
        """
        vertices = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = np.radians(angle_deg)
            x_vertex = self.x + self.coverage_radius * np.cos(angle_rad)
            y_vertex = self.y + self.coverage_radius * np.sin(angle_rad)
            vertices.append((x_vertex, y_vertex))
        return vertices

    def _calculate_hexagon_area(self):
        """
        Calculate the area of the regular hexagon.
        Area = (3√3/2) * r² where r is the circumradius
        """
        return (3 * np.sqrt(3) / 2) * (self.coverage_radius ** 2)

    def is_point_in_coverage(self, ue_x, ue_y):
        """
        Check if a point (UE) is inside the hexagonal coverage area.
        Uses ray casting algorithm for point-in-polygon.
        """
        distance = self.distance_to_ue(ue_x, ue_y)
        if distance > self.coverage_radius:
            return False

        x, y = ue_x, ue_y
        inside = False

        for i in range(len(self.vertices)):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % len(self.vertices)]

            if self._point_on_line_segment(x, y, x1, y1, x2, y2):
                return True

            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside

        return inside

    def _point_on_line_segment(self, x, y, x1, y1, x2, y2):
        """
        Check if a point lies on a line segment.
        """
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)

        if x < min_x - 1e-10 or x > max_x + 1e-10 or y < min_y - 1e-10 or y > max_y + 1e-10:
            return False

        cross_product = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
        if abs(cross_product) > 1e-10:
            return False

        return True

    def get_coverage_boundaries(self):
        """
        Get the bounding box of the coverage area.
        Returns:
            (x_min, x_max, y_min, y_max)
        """
        x_coords = [v[0] for v in self.vertices]
        y_coords = [v[1] for v in self.vertices]
        return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

    def distance_to_ue(self, ue_x, ue_y):
        """
        Calculate Euclidean distance from this gNodeB to a UE.
        """
        return np.sqrt((self.x - ue_x) ** 2 + (self.y - ue_y) ** 2)

    # ------------------------------------------------------------------
    # Radio helpers
    # ------------------------------------------------------------------

    def get_rb_bandwidth_hz(self):
        """
        Approximate bandwidth of one PRB.
        """
        return 180e3

    def get_noise_power_dbm(self, rb_bandwidth_hz=None):
        """
        Thermal noise power over one RB in dBm.
        N = -174 dBm/Hz + 10log10(BW) + NF
        """
        if rb_bandwidth_hz is None:
            rb_bandwidth_hz = self.get_rb_bandwidth_hz()
        return -174 + 10 * np.log10(rb_bandwidth_hz) + self.noise_figure_db

    def get_received_power_dbm(self, ue_x, ue_y):
        """
        Received power at UE position using a more realistic macro-cell path-loss model.
        Returns -inf if UE is out of coverage.
        """
        if not self.is_point_in_coverage(ue_x, ue_y):
            return -np.inf

        d_m = max(self.distance_to_ue(ue_x, ue_y), 10.0)  # avoid singularity / unrealistically close range
        d_km = d_m / 1000.0
        f_ghz = self.center_frequency_hz / 1e9

        # Urban macro style path loss (close to your older channel model)
        # 128.1 + 37.6 log10(d_km) is commonly used around 2 GHz urban macro.
        # Add a mild frequency correction so 3.5 GHz is slightly harsher.
        path_loss_db = 128.1 + 37.6 * np.log10(d_km) + 20.0 * np.log10(f_ghz / 2.0)

        # Optional log-normal shadowing. The dense SUMO scenario uses the
        # CSV fading traces in MultiGNBWrapper, so this defaults to 0.
        shadowing_db = (
            np.random.normal(0.0, self.shadowing_std_db)
            if self.shadowing_std_db > 0.0
            else 0.0
        )

        # Optional minimum coupling loss clamp
        path_loss_db = max(path_loss_db + shadowing_db, 70.0)

        return self.tx_power_dbm - path_loss_db

    def get_received_power_watts(self, ue_x, ue_y):
        """
        Received power in watts.
        """
        p_dbm = self.get_received_power_dbm(ue_x, ue_y)
        if not np.isfinite(p_dbm):
            return 0.0
        return 10 ** ((p_dbm - 30) / 10.0)

    def uses_same_carrier(self, other_nodeb):
        """
        Check if two gNodeBs use the same carrier.
        """
        return self.carrier_id == other_nodeb.carrier_id

    def get_ue_signal_strength(self, ue_x, ue_y):
        """
        Calculate normalized signal strength for a UE based on distance.
        Returns 0 if out of coverage.
        """
        if not self.is_point_in_coverage(ue_x, ue_y):
            return 0.0

        distance = self.distance_to_ue(ue_x, ue_y)
        normalized_distance = distance / self.coverage_radius
        signal_strength = 1.0 / (1.0 + normalized_distance ** 2)

        return signal_strength

    def get_ue_snr(self, ue_x, ue_y, rb_bandwidth_hz=None):
        """
        Calculate SNR for a UE in dB using received power and thermal noise.
        """
        p_rx_dbm = self.get_received_power_dbm(ue_x, ue_y)
        if not np.isfinite(p_rx_dbm):
            return -np.inf

        noise_dbm = self.get_noise_power_dbm(rb_bandwidth_hz=rb_bandwidth_hz)
        return p_rx_dbm - noise_dbm

    # ------------------------------------------------------------------
    # Geometry / overlap
    # ------------------------------------------------------------------

    def get_overlapping_coverage(self, other_nodeb):
        """
        Check if coverage areas overlap with another gNodeB.
        """
        distance = self.distance_to_ue(other_nodeb.x, other_nodeb.y)
        return distance < (self.coverage_radius + other_nodeb.coverage_radius)

    def get_overlap_area_estimate(self, other_nodeb):
        """
        Rough estimate of overlap area with another gNodeB.
        Uses circle-circle overlap as an approximation.
        """
        distance = self.distance_to_ue(other_nodeb.x, other_nodeb.y)

        if distance == 0:
            return min(self.coverage_area, other_nodeb.coverage_area)

        if distance >= (self.coverage_radius + other_nodeb.coverage_radius):
            return 0.0

        if distance <= abs(self.coverage_radius - other_nodeb.coverage_radius):
            return min(self.coverage_area, other_nodeb.coverage_area)

        r1 = self.coverage_radius
        r2 = other_nodeb.coverage_radius

        part1 = r1 ** 2 * np.arccos((distance ** 2 + r1 ** 2 - r2 ** 2) / (2 * distance * r1))
        part2 = r2 ** 2 * np.arccos((distance ** 2 + r2 ** 2 - r1 ** 2) / (2 * distance * r2))
        part3 = 0.5 * np.sqrt(
            (-distance + r1 + r2)
            * (distance + r1 - r2)
            * (distance - r1 + r2)
            * (distance + r1 + r2)
        )

        overlap_area = part1 + part2 - part3
        return max(0, overlap_area)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_coverage(
        self,
        ax=None,
        show_slices=False,
        color='lightblue',
        alpha=0.3,
        edge_color='blue',
        linewidth=2,
        show_sector_ids=False,
        set_limits=False,
    ):
        """
        Plot the hexagonal coverage area with customizable visualization options.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon

            if ax is None:
                _, ax = plt.subplots()

            hex_patch = Polygon(
                self.vertices,
                closed=True,
                edgecolor=edge_color if edge_color != 'none' else 'none',
                facecolor=color,
                alpha=alpha,
                linewidth=linewidth
            )
            ax.add_patch(hex_patch)

            marker_color = 'red' if edge_color != 'none' else 'blue'
            ax.plot(
                self.x,
                self.y,
                '^',
                color=marker_color,
                markersize=10,
                label=f'gNodeB {self.id}',
                zorder=5
            )

            if show_slices and self.n_slices_l1 > 1:
                self._draw_slice_divisions(ax)

            if show_sector_ids:
                self._draw_sector_ids(ax)

            if edge_color != 'none':
                ax.annotate(
                    f'R={self.coverage_radius}',
                    xy=(self.x, self.y + self.coverage_radius),
                    xytext=(self.x + 10, self.y + self.coverage_radius + 10),
                    arrowprops=dict(arrowstyle='->'),
                    fontsize=8
                )

            ax.set_aspect('equal')

            if set_limits:
                x_coords = [v[0] for v in self.vertices]
                y_coords = [v[1] for v in self.vertices]
                margin = self.coverage_radius * 0.1
                ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
                ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)

            return ax

        except ImportError:
            print("Matplotlib not available for visualization")
            return None

    def _draw_slice_divisions(self, ax):
        """
        Draw divisions between slices in the hexagon.
        """
        try:
            n_slices = self.n_slices_l1
            if n_slices <= 1:
                return

            angle_step = 360 / n_slices
            for i in range(n_slices):
                angle_rad = np.radians(i * angle_step)
                x_end = self.x + self.coverage_radius * np.cos(angle_rad)
                y_end = self.y + self.coverage_radius * np.sin(angle_rad)
                ax.plot([self.x, x_end], [self.y, y_end], 'k--', alpha=0.5, linewidth=1)
        except Exception as e:
            print(f"Error drawing slice divisions: {e}")

    def _draw_sector_ids(self, ax):
        """
        Draw sector/slice IDs on the hexagon.
        """
        try:
            n_slices = self.n_slices_l1
            angle_step = 360 / n_slices if n_slices > 0 else 360

            for i in range(n_slices):
                angle_rad = np.radians(i * angle_step + angle_step / 2)
                text_radius = self.coverage_radius * 0.65
                x_text = self.x + text_radius * np.cos(angle_rad)
                y_text = self.y + text_radius * np.sin(angle_rad)
                ax.text(
                    x_text,
                    y_text,
                    f'S{i}',
                    ha='center',
                    va='center',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='circle', facecolor='white', alpha=0.7)
                )
        except Exception as e:
            print(f"Error drawing sector IDs: {e}")

    # ------------------------------------------------------------------
    # Simulation API
    # ------------------------------------------------------------------

    def reset(self):
        self.steps = 0
        self._a3_counters.clear()
        for slice_l1 in self.slices_l1:
            slice_l1.reset()
        return self.get_state()

    def get_n_variables(self):
        n_variables = 0
        for slice_l1 in self.slices_l1:
            n_variables += slice_l1.get_n_variables()
        return n_variables

    def reset_info(self):
        """
        Reset the info of the L1 slices for SLA assessment.
        """
        for l1 in self.slices_l1:
            l1.reset_info()

    def slot(self):
        """
        Run the system for one time-slot.
        """
        for slice_l1 in self.slices_l1:
            slice_l1.slot()

    def get_state(self):
        state = np.array([self.x, self.y], dtype=float)
        for l1 in self.slices_l1:
            state = np.concatenate((state, l1.get_state()), axis=None)
        return state

    def get_info(self, violations=0, SLA_labels=0):
        return {
            'l1_info': [l1.get_info() for l1 in self.slices_l1],
            'SLA_labels': SLA_labels,
            'violations': violations,
            'n_prbs': [l1.n_prbs for l1 in self.slices_l1]
        }

    def compute_reward(self):
        """
        Check if the SLA is fulfilled for each slice.
        """
        SLA_labels = np.zeros(self.n_slices_l1, dtype=int)
        violations = np.zeros(self.n_slices_l1, dtype=int)
        for i, l1 in enumerate(self.slices_l1):
            SLA_labels[i], violations[i] = l1.compute_reward()
        return SLA_labels, violations

    def step(self, action):
        """
        Move one simulation step forward using the selected action.
        Each step consists of a number of time slots.
        """
        self.reset_info()

        if action is None or len(action) != len(self.slices_l1):
            print('The action must contain as many elements as slices!')
            return self.get_state(), self.get_info()

        if np.sum(action) > self.n_prbs:
            print('Total PRBs in action exceed available PRBs!')
            return self.get_state(), self.get_info()

        i_prb = 0
        for slice_l1, prbs in zip(self.slices_l1, action):
            slice_l1.set_prbs(i_prb, int(prbs))
            i_prb += int(prbs)

        for _ in range(self.slots_per_step):
            self.slot()

        state = self.get_state()
        SLA_labels, violations = self.compute_reward()
        info = self.get_info(SLA_labels=SLA_labels, violations=violations)

        self.steps += 1
        return state, info

    def __repr__(self):
        return (
            f'NodeB {self.id} at ({self.x:.2f}, {self.y:.2f}), '
            f'radius={self.coverage_radius}, '
            f'carrier_id={self.carrier_id}, '
            f'f={self.center_frequency_hz / 1e9:.2f}GHz, '
            f'bw={self.bandwidth_hz / 1e6:.1f}MHz'
        )

    def get_l1_slice_for_type(self, slice_type: str):
        wanted = (slice_type or "eMBB").upper()

        for s in self.slices_l1:
            st = getattr(s, "type", "").upper()
            if wanted == "URLLC" and st == "URLLC":
                return s
            if wanted == "MMTC" and st == "MMTC":
                return s
            if wanted == "EMBB" and st == "EMBB":
                return s

        # fallback: use eMBB slice for unknown broadband-like traffic
        for s in self.slices_l1:
            if getattr(s, "type", "").upper() == "EMBB":
                return s

        return None

    def attach_ue(self, ue):
        if str(getattr(ue, "slice_type", "") or "").upper() == "MMTC":
            # SliceL1mMTC models aggregate NB-IoT devices, not mobile UE
            # queues. Local A3 training still needs mobile mMTC UEs for
            # handover decisions, so attach them to a mobile-capable L1 while
            # preserving ue.slice_type == mMTC for slice-load accounting.
            for s in self.slices_l1:
                if getattr(s, "type", "").upper() in {"EMBB", "URLLC"}:
                    s.add_users([ue])
                    return True
            return False

        l1 = self.get_l1_slice_for_type(getattr(ue, "slice_type", "eMBB"))
        if l1 is None:
            raise ValueError(f"No compatible L1 slice in gNB {self.id} for UE slice_type={ue.slice_type}")
        l1.add_users([ue])
        return True

    def detach_ue(self, ue_id: int):
        for s in self.slices_l1:
            if hasattr(s, "extract_users"):
                s.extract_users([ue_id])

    def count_attached_rl_ues(self) -> int:
        total = 0
        for s in self.slices_l1:
            if hasattr(s, "ues"):
                total += len(s.ues)
        return total
if __name__ == '__main__':
    from numpy.random import default_rng
    from itertools import count
    import matplotlib.pyplot as plt

    from ran.slice_l1 import SliceL1eMBB, SliceL1mMTC
    from ran.slice_ran import SliceRANeMBB, SliceRANmMTC
    from ran.channel_models import SINRSelectiveFading, MCSCodeset
    from ran.schedulers import ProportionalFair

    rng = default_rng(seed=42)

    # -----------------------------
    # Shared simulation parameters
    # -----------------------------
    slots_per_step = 50
    slot_length = 1e-3
    n_prbs = 100

    CBR_description = {
        'lambda': 2.0 / 60.0,
        't_mean': 30.0,
        'bit_rate': 500000
    }

    VBR_description = {
        'lambda': 5.0 / 60.0,
        't_mean': 30.0,
        'p_size': 1000,
        'b_size': 500,
        'b_rate': 1
    }

    SLA_embb = {
        'cbr_th': 10e6,
        'cbr_prb': 20,
        'cbr_queue': 10e4,
        'vbr_th': 15e6,
        'vbr_prb': 30,
        'vbr_queue': 15e4
    }

    state_variables_embb = [
        'cbr_traffic', 'cbr_th', 'cbr_prb', 'cbr_queue', 'cbr_snr',
        'vbr_traffic', 'vbr_th', 'vbr_prb', 'vbr_queue', 'vbr_snr'
    ]

    time_per_step = slots_per_step * slot_length
    norm_const_embb = {
        'cbr_traffic': 5e6 * time_per_step,
        'cbr_th': 10e6 * time_per_step,
        'cbr_prb': 25 * slots_per_step,
        'cbr_queue': 10e4 * slots_per_step,
        'cbr_snr': 35 * slots_per_step,
        'vbr_traffic': 5e6 * time_per_step,
        'vbr_th': 10e6 * time_per_step,
        'vbr_prb': 35 * slots_per_step,
        'vbr_queue': 10e4 * slots_per_step,
        'vbr_snr': 35 * slots_per_step
    }

    MTC_description = {
        'n_devices': 1000,
        'repetition_set': [2, 4, 8, 16, 32, 64, 128],
        'period_set': [1000, 50000, 10000, 15000, 20000, 25000, 50000, 100000]
    }

    state_variables_mmtc = ['devices', 'avg_rep', 'delay']
    SLA_mmtc = {'delay': 300}
    norm_const_mmtc = {
        'devices': 100 * slots_per_step,
        'avg_rep': 100 * slots_per_step,
        'delay': 100 * slots_per_step
    }

    # -----------------------------
    # Shared channel/scheduler
    # -----------------------------
    snr_generator = SINRSelectiveFading(rng, 'macro_cell_urban_2GHz', n_prbs=n_prbs)
    mcs_codeset = MCSCodeset()
    scheduler = ProportionalFair(mcs_codeset)
    user_counter = count()

    # -----------------------------
    # gNodeB 1 slices
    # -----------------------------
    slice_ran_embb_1 = SliceRANeMBB(
        rng, user_counter, 0,
        SLA_embb, CBR_description, VBR_description,
        state_variables_embb, norm_const_embb,
        slots_per_step, slot_length=slot_length
    )
    slice_l1_embb_1 = SliceL1eMBB(rng, snr_generator, 50, [slice_ran_embb_1], scheduler)

    slice_ran_mmtc_1 = SliceRANmMTC(
        rng, 1, SLA_mmtc, MTC_description,
        state_variables_mmtc, norm_const_mmtc,
        slots_per_step
    )
    slice_l1_mmtc_1 = SliceL1mMTC(50, [slice_ran_mmtc_1])

    gnb1_slices = [slice_l1_embb_1, slice_l1_mmtc_1]

    # -----------------------------
    # gNodeB 2 slices
    # -----------------------------
    slice_ran_embb_2 = SliceRANeMBB(
        rng, user_counter, 2,
        SLA_embb, CBR_description, VBR_description,
        state_variables_embb, norm_const_embb,
        slots_per_step, slot_length=slot_length
    )
    slice_l1_embb_2 = SliceL1eMBB(rng, snr_generator, 50, [slice_ran_embb_2], scheduler)

    gnb2_slices = [slice_l1_embb_2]

    # -----------------------------
    # Create overlapping gNodeBs
    # -----------------------------
    gnb1 = NodeB(
        id=1,
        x=100,
        y=100,
        slices_l1=gnb1_slices,
        slots_per_step=slots_per_step,
        n_prbs=n_prbs,
        coverage_radius=300,
        slot_length=slot_length,
        carrier_id=0,
        center_frequency_hz=3.5e9,
        bandwidth_hz=20e6,
        tx_power_dbm=30.0,
        noise_figure_db=7.0
    )

    gnb2 = NodeB(
        id=2,
        x=340,
        y=180,
        slices_l1=gnb2_slices,
        slots_per_step=slots_per_step,
        n_prbs=n_prbs,
        coverage_radius=300,
        slot_length=slot_length,
        carrier_id=0,   # different carrier first
        center_frequency_hz=3.7e9,
        bandwidth_hz=20e6,
        tx_power_dbm=30.0,
        noise_figure_db=7.0
    )

    print("=" * 70)
    print("NODEB OBJECTS")
    print("=" * 70)
    print(gnb1)
    print(gnb2)

    # -----------------------------
    # Test points
    # -----------------------------
    test_points = [
        (100, 100, "Center gNB1"),
        (340, 180, "Center gNB2"),
        (220, 140, "Overlap region"),
        (20, 20, "Edge / outside"),
        (260, 120, "Near overlap"),
    ]

    print("\n" + "=" * 70)
    print("COVERAGE / POWER / SNR TEST (DIFFERENT CARRIERS)")
    print("=" * 70)

    for x, y, label in test_points:
        in_1 = gnb1.is_point_in_coverage(x, y)
        in_2 = gnb2.is_point_in_coverage(x, y)

        p1 = gnb1.get_received_power_dbm(x, y)
        p2 = gnb2.get_received_power_dbm(x, y)

        snr1 = gnb1.get_ue_snr(x, y)
        snr2 = gnb2.get_ue_snr(x, y)

        print(f"\nPoint: {label} -> ({x}, {y})")
        print(f"  In gNB1 coverage: {in_1}")
        print(f"  In gNB2 coverage: {in_2}")
        print(f"  gNB1 received power: {p1:.2f} dBm")
        print(f"  gNB2 received power: {p2:.2f} dBm")
        print(f"  gNB1 SNR: {snr1:.2f} dB")
        print(f"  gNB2 SNR: {snr2:.2f} dB")

        if in_1 and in_2:
            best = "gNB1" if p1 > p2 else "gNB2"
            print(f"  Best server in overlap: {best}")

    # -----------------------------
    # Overlap / carrier checks
    # -----------------------------
    print("\n" + "=" * 70)
    print("OVERLAP / CARRIER CHECK")
    print("=" * 70)
    print("Geometric overlap:", gnb1.get_overlapping_coverage(gnb2))
    print(f"Overlap area estimate: {gnb1.get_overlap_area_estimate(gnb2):.2f} m^2")
    print("Same carrier:", gnb1.uses_same_carrier(gnb2))

    # -----------------------------
    # Same-carrier test
    # -----------------------------
    gnb2.carrier_id = 0

    print("\n" + "=" * 70)
    print("SAME-CARRIER CHECK")
    print("=" * 70)
    print("Same carrier after update:", gnb1.uses_same_carrier(gnb2))
    print("This means interference should be considered in wrapper-level SINR later.")

    # -----------------------------
    # Quick NodeB step test
    # -----------------------------
    print("\n" + "=" * 70)
    print("NODE STEP TEST")
    print("=" * 70)

    state1_before = gnb1.get_state()
    print("gNB1 state size before step:", len(state1_before))

    action1 = [50, 50]   # 2 slices in gNB1
    state1_after, info1 = gnb1.step(action1)
    print("gNB1 state size after step:", len(state1_after))
    print("gNB1 info keys:", list(info1.keys()))

    action2 = [100]      # 1 slice in gNB2
    state2_after, info2 = gnb2.step(action2)
    print("gNB2 state size after step:", len(state2_after))
    print("gNB2 info keys:", list(info2.keys()))

    # -----------------------------
    # Visualization
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 7))

    gnb1.visualize_coverage(
        ax=ax,
        show_slices=True,
        color='lightblue',
        alpha=0.25,
        edge_color='blue',
        linewidth=2,
        show_sector_ids=True
    )

    gnb2.visualize_coverage(
        ax=ax,
        show_slices=True,
        color='lightgreen',
        alpha=0.25,
        edge_color='green',
        linewidth=2,
        show_sector_ids=True
    )

    for x, y, label in test_points:
        in_1 = gnb1.is_point_in_coverage(x, y)
        in_2 = gnb2.is_point_in_coverage(x, y)

        if in_1 and in_2:
            color = 'purple'
        elif in_1 or in_2:
            color = 'orange'
        else:
            color = 'red'

        ax.plot(x, y, 'o', color=color, markersize=7, markeredgecolor='black')
        ax.text(x + 5, y + 5, label, fontsize=8)

    # Global axis limits
    all_x = [v[0] for v in gnb1.vertices] + [v[0] for v in gnb2.vertices]
    all_y = [v[1] for v in gnb1.vertices] + [v[1] for v in gnb2.vertices]
    margin = 80
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_aspect('equal')
    ax.set_title("Two overlapping gNodeBs - geometry and test points")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("Next step: compute wrapper-level interference and SINR in overlap areas.")
