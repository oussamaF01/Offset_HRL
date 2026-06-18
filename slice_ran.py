#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

Classes:

UE
SliceRANmMTC
SliceRANeMBB

"""
DEBUG = True
CBR = 0
VBR = 1

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
from traffic_generators import VbrSource, CbrSource
from channel_models import generate_xy


@dataclass
class Packet:
    """Individual packet with arrival timing information."""
    bits: int
    arrival_step: int
    arrival_time_s: float
    packet_id: int = 0

    def __repr__(self):
        return f"Packet(id={self.packet_id}, bits={self.bits}, step={self.arrival_step})"


class UE:
    '''
    eMBB/URLLC UE with mobility, packet queue, radio, and serving-cell information.
    '''
    packet_queue = None
    hol_delay_s = 0.0

    def __init__(
        self,
        id,
        slice_ran_id,
        traffic_source,
        type,
        x=0,
        y=0,
        vx=0,
        vy=0,
        window=50,
        slot_length=1e-3,
        slice_type=None,
        buffer_size=np.inf
    ):
        self.id = id
        self.slice_ran_id = slice_ran_id
        self.slice_type = slice_type
        self.traffic_source = traffic_source
        self.type = type

        # Mobility
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

        # Serving / handover
        self.serving_gnb = None
        self.target_gnb = None
        self.connected = True
        self.ho_pending = False
        self.ho_candidate = None
        self.ho_counter = 0

        # Throughput smoothing
        self.th = 0
        self.b = 1 / window
        self.a = 1 - self.b

        # Queue / traffic - packet level.
        self.packet_queue: Deque[Packet] = deque()
        self.queue = 0
        self.buffer_size = buffer_size
        self.slot_length = slot_length
        self.new_bits = 0
        self.bits = 0
        self.scheduled_bits = 0
        self.useful_prbs = 0
        self.wasted_prbs = 0
        self.dropped_bits = 0
        self.dropped_bits_step = 0
        self.total_bits_arrived = 0
        self.total_bits_dropped = 0

        self._next_packet_id = 0
        self._step_counter = 0
        self._last_packet_arrival_step = 0

        # Delay. wait_time is kept as a legacy slot-count view.
        self.wait_time = 0
        self.hol_delay_s = 0.0
        self.max_hol_delay_s = 0.0
        self.mean_packet_delay_s = 0.0
        self._delay_samples: Deque[float] = deque(maxlen=100)

        # Radio variables
        self.snr = 0
        self.e_snr = 0
        self.sinr = 0
        self.e_sinr = 0
        self.prbs = 0
        self.p = 0
        self.mcs = None
        self.spectral_efficiency = 0.0

        # Debug radio metrics
        self.serving_power_dbm = -np.inf
        self.interference_dbm = -np.inf
        self.noise_dbm = -np.inf

    def set_global_step(self, step: int):
        """Set current global step for packet timing."""
        self._step_counter = max(int(step), int(self._step_counter))

    def get_current_time_s(self) -> float:
        return float(self._step_counter) * float(self.slot_length)

    def _update_hol_delay(self) -> float:
        if not self.packet_queue:
            return 0.0

        hol_delay = self.get_current_time_s() - self.packet_queue[0].arrival_time_s
        hol_delay = max(0.0, float(hol_delay))
        self.max_hol_delay_s = max(float(self.max_hol_delay_s), hol_delay)
        return hol_delay

    def _update_mean_delay(self):
        if self._delay_samples:
            self.mean_packet_delay_s = sum(self._delay_samples) / len(self._delay_samples)

    def get_delay_s(self) -> float:
        return float(self.hol_delay_s)

    def get_max_delay_s(self) -> float:
        return float(self.max_hol_delay_s)

    def estimate_snr(self, snr):
        self.snr = snr
        m = np.mean(snr)
        self.e_snr = float(m) if np.isfinite(m) else -np.inf

    def estimate_sinr(self, sinr):
        self.sinr = sinr
        self.e_sinr = round(np.mean(sinr)) if hasattr(sinr, "__len__") else float(sinr)

    def traffic_step(self):
        """Generate new traffic as timestamped packets."""
        self.new_bits = self.traffic_source.step()
        self.dropped_bits_step = 0
        self.total_bits_arrived += self.new_bits

        if self.new_bits > 0:
            packet = Packet(
                bits=int(self.new_bits),
                arrival_step=int(self._step_counter),
                arrival_time_s=self.get_current_time_s(),
                packet_id=int(self._next_packet_id),
            )
            self._next_packet_id += 1
            self._last_packet_arrival_step = int(self._step_counter)
            self.packet_queue.append(packet)
            self.queue += self.new_bits
            if len(self.packet_queue) == 1:
                self.hol_delay_s = 0.0

        if self.queue > self.buffer_size:
            overflow = self.queue - self.buffer_size
            while overflow > 0 and self.packet_queue:
                dropped = self.packet_queue.popleft()
                overflow -= dropped.bits
                self.dropped_bits_step += dropped.bits
                self.dropped_bits += dropped.bits
                self.total_bits_dropped += dropped.bits
                self.queue -= dropped.bits
                self._delay_samples.append(
                    self.get_current_time_s() - dropped.arrival_time_s
                )
            self.queue = min(self.queue, self.buffer_size)

        self.hol_delay_s = self._update_hol_delay()
        self.wait_time = min(self.hol_delay_s / max(float(self.slot_length), 1e-12), 1000.0)

    def transmission_step(self, received):
        """Serve queued packet bits FIFO and record completed packet latency."""
        self.scheduled_bits = max(float(self.bits), 0.0)
        if received:
            served_bits = min(float(self.queue), self.scheduled_bits)
            self.bits = served_bits

            remaining = int(served_bits)
            while remaining > 0 and self.packet_queue:
                packet = self.packet_queue[0]
                if packet.bits <= remaining:
                    served_packet = self.packet_queue.popleft()
                    remaining -= served_packet.bits
                    packet_delay = self.get_current_time_s() - served_packet.arrival_time_s
                    self._delay_samples.append(packet_delay)
                else:
                    packet.bits -= remaining
                    remaining = 0
        else:
            self.bits = 0

        self.queue = max(self.queue - self.bits, 0)
        self.th = self.a * self.th + self.b * self.bits / self.slot_length
        self.hol_delay_s = self._update_hol_delay()
        self._update_mean_delay()
        self.wait_time = min(self.hol_delay_s / max(float(self.slot_length), 1e-12), 1000.0)

    def update_position(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

    def get_state(self):
        return np.array([
            self.queue,
            self.th,
            self.e_sinr if self.e_sinr != 0 else self.e_snr,
            self.bits,
            self.x,
            self.y
        ], dtype=float)

    def __repr__(self):
        return f'UE {self.id} (slice={self.slice_type}, delay={self.hol_delay_s:.3f}s, queue={self.queue})'
class MTCdevice:
    def __init__(self, id, repetitions, slice_ran_id):
        self.id = id
        self.repetitions = repetitions
        self.slice_ran_id = slice_ran_id
    def __repr__(self):
        return 'MTC {}'.format(self.id)





"""
this is the same as the old similator 
"""
class SliceRANmMTC:
    '''
    Generates message arrivals at the mMTC devices
    according to the characteristics defined in MTC_description:
    - n_devices: total number of devices
    - repetition_set: possible repetitions
    - period_set: possible times between message arrivals
    '''
    def __init__(self, rng, id, SLA, MTCdescription, state_variables, norm_const, slots_per_step):
        self.type = 'mMTC'
        self.rng = rng
        self.id = id
        self.SLA = SLA
        self.state_variables = state_variables # ['devices', 'avg_rep', 'delay']
        self.norm_const = norm_const # 100 all
        self.slots_per_step = slots_per_step

        self.n_devices = MTCdescription['n_devices']
        self.repetition_set = MTCdescription['repetition_set']
        self.period_set = MTCdescription['period_set']

        self.reset()

    def reset(self):
        self.reset_state()
        self.reset_info()
        self.period = np.ones((self.n_devices), dtype=int)
        self.t_to_arrival = np.zeros((self.n_devices), dtype=int)
        self.devices = []
        for i in range(self.n_devices):
            repetitions = self.rng.choice(self.repetition_set)
            self.period[i] = self.rng.choice(self.period_set)
            self.t_to_arrival[i] = 1 + int(self.rng.integers(0, self.period[i]))
            self.devices.append(MTCdevice(i, repetitions, self.id))

    def slot(self):
        self.slot_counter += 1

        # advance time
        self.t_to_arrival -= 1

        # arrivals
        arrival_list = []
        arrivals = self.t_to_arrival == 0
        indices = np.where(arrivals)

        # print('indices = {}'.format(indices))
        for i in indices[0]:
            arrival_list.append(self.devices[i])

        # prepare for next arrival (deterministic inter arrival time)
        self.t_to_arrival[arrivals] = self.period[arrivals]

        return arrival_list, []

    def reset_info(self):
        self.info = {'delay': 0, 'avg_rep': 0, 'devices': 0}
        self.slot_counter = 0

    def reset_state(self):
        self.state = np.full((len(self.state_variables)), 0, dtype = float)

    def get_n_variables(self):
        return len(self.state_variables)

    def get_state(self):
        '''convert the info into a normalized vector'''
        for i, var in enumerate(self.state_variables):
            self.state[i] = self.info[var] / self.norm_const[var]        
        return self.state


    def update_info(self, delay, avg_rep, devices):
        self.info['delay'] += delay
        self.info['avg_rep'] += avg_rep
        self.info['devices'] += devices
        

    def compute_reward(self):
        '''assesses SLA violations'''
        SLA_fulfilled = self.info['delay']/self.slots_per_step < self.SLA['delay']
        return not(SLA_fulfilled)

class SliceRANeMBB:
    '''
    Generates arrivals and departures of eMBB ues.
    There are two traffic types: CRB (GBR) and VBR (non-GBR)
    CBR traffic parameters are given in CBR_description
    VBR traffic parameters are given in VBR_description
    '''
    def __init__(self, rng, user_counter, id, SLA, CBR_description, VBR_description, state_variables, norm_const, slots_per_step, slot_length = 1e-3):
        self.type = 'eMBB'
        self.rng = rng
        self.user_counter = user_counter
        self.id = id
        self.slot_length = slot_length
        self.slots_per_step = slots_per_step
        self.observation_time = slots_per_step * slot_length
        self.SLA = SLA # service level agreement description
        self.state_variables = state_variables
        self.norm_const = norm_const

        self.cbr_arrival_rate = CBR_description['lambda']
        self.cbr_mean_time = CBR_description['t_mean']
        self.cbr_bit_rate = CBR_description['bit_rate']

        self.vbr_arrival_rate = VBR_description['lambda']
        self.vbr_mean_time = VBR_description['t_mean']
        self.vbr_source_data = {
            'packet_size': VBR_description['p_size'],
            'burst_size': VBR_description['b_size'],
            'burst_rate':VBR_description['b_rate']
        }
        self.reset()

    def reset(self):
        self.slot_counter = 0
        self.remaining_time = {}
        self.cbr_steps_next_arrival = 0
        self.vbr_steps_next_arrival = 0
        self.vbr_ues = {}
        self.cbr_ues = {}
        self.reset_state()
        self.reset_info()

    def get_n_variables(self):
        return len(self.state_variables)

    def cbr_cac(self):
        '''Admission control for CBR users'''
        slots = max(self.slot_counter,1)
        time = slots * self.slot_length
        cbr_prb = self.info['cbr_prb'] / slots
        cbr_th = self.info['cbr_th'] / time
        if cbr_prb >= self.SLA['cbr_prb'] or cbr_th >= self.SLA['cbr_th']:
            return False
        return True

    def cbr_arrivals(self):
        if self.cbr_steps_next_arrival == 0:
            # generate next arrival
            inter_arrival_time = self.rng.exponential(1.0 / self.cbr_arrival_rate)
            inter_arrival_time = np.rint(inter_arrival_time / self.slot_length)
            self.cbr_steps_next_arrival = inter_arrival_time

            if self.cbr_cac(): # check admission control
                # generate new user
                ue_id = next(self.user_counter)
                cbr_source = CbrSource(bit_rate = self.cbr_bit_rate)
                ue = UE(ue_id, self.id, cbr_source, CBR)
                # Assign coordinates and velocity
                ue.x, ue.y = generate_xy(self.rng)
                ue.vx = self.rng.normal(0, 5)  # random velocity in x (km/h)
                ue.vy = self.rng.normal(0, 5)  # random velocity in y (km/h)
                self.cbr_ues[ue_id] = ue

                # generate holding time
                holding_time = self.rng.exponential(self.cbr_mean_time)
                holding_time = np.rint(holding_time / self.slot_length)
                self.remaining_time[ue_id] = holding_time

                return [ue] # return user
        else:
            self.cbr_steps_next_arrival -= 1    
        return []

    def vbr_arrivals(self):
        if self.vbr_steps_next_arrival == 0:
            # create new vbr user
            ue_id = next(self.user_counter)
            vbr_source = VbrSource(**self.vbr_source_data)
            ue = UE(ue_id, self.id, vbr_source, VBR)
            # Assign coordinates and velocity
            ue.x, ue.y = generate_xy(self.rng)
            ue.vx = self.rng.normal(0, 5)  # random velocity in x (km/h)
            ue.vy = self.rng.normal(0, 5)  # random velocity in y (km/h)
            self.vbr_ues[ue_id] = ue

            # generate holding time
            holding_time = self.rng.exponential(self.vbr_mean_time)
            holding_time = np.rint(holding_time / self.slot_length)
            self.remaining_time[ue_id] = holding_time

            # generate next arrival
            inter_arrival_time = self.rng.exponential(1.0 / self.vbr_arrival_rate)
            inter_arrival_time = np.rint(inter_arrival_time / self.slot_length)
            self.vbr_steps_next_arrival = inter_arrival_time
            return [ue]
        else:
            self.vbr_steps_next_arrival -= 1
            return []

    def departures(self):
        departures = []
        current_ids = list(self.remaining_time.keys())
        for id in current_ids:
            self.remaining_time[id] -= 1
            if self.remaining_time[id] == 0:
                departures.append(id)
                del self.remaining_time[id] # delete timer
                self.vbr_ues.pop(id, None) # delete ue if here
                self.cbr_ues.pop(id, None) # or here    
        return departures   

    def slot(self):
        self.slot_counter += 1
        arrivals = self.cbr_arrivals()
        arrivals.extend(self.vbr_arrivals())
        departures = self.departures()
        return arrivals, departures

    def reset_info(self):
        self.info = {'cbr_traffic': 0, 'cbr_th': 0, 'cbr_prb': 0, 'cbr_queue':0, 'cbr_snr': 0,\
                    'vbr_traffic': 0, 'vbr_th': 0, 'vbr_prb': 0, 'vbr_queue': 0, 'vbr_snr': 0}
        self.slot_counter = 0

    def reset_state(self):
        self.state = np.full((len(self.state_variables)), 0, dtype = float)
    
    def update_info(self):
        queue = 0
        snr = 0
        n = 0
        for ue in self.cbr_ues.values():
            self.info['cbr_traffic'] += ue.new_bits
            self.info['cbr_th'] += ue.bits
            self.info['cbr_prb'] += ue.prbs
            queue += ue.queue
            snr += ue.e_snr
            n += 1
        n = max(n,1)
        self.info['cbr_queue'] += queue/n
        self.info['cbr_snr'] += snr/n

        queue = 0
        snr = 0
        n = 0
        for ue in self.vbr_ues.values():
            self.info['vbr_traffic'] += ue.new_bits
            self.info['vbr_th'] += ue.bits
            self.info['vbr_prb'] += ue.prbs
            queue += ue.queue
            snr += ue.e_snr
            n += 1
        n = max(n,1)
        self.info['vbr_queue'] += queue/n
        self.info['vbr_snr'] += snr/n

    def compute_reward(self):
        '''assesses SLA violations'''
        cbr_th = self.info['cbr_th']/self.observation_time > self.SLA['cbr_th']
        cbr_prb = self.info['cbr_prb']/self.slots_per_step > self.SLA['cbr_prb']
        cbr_queue = self.info['cbr_queue']/self.slots_per_step < self.SLA['cbr_queue']
        vbr_th = self.info['vbr_th']/self.observation_time > self.SLA['vbr_th']
        vbr_prb = self.info['vbr_prb']/self.slots_per_step > self.SLA['vbr_prb']
        vbr_queue = self.info['vbr_queue']/self.slots_per_step < self.SLA['vbr_queue']
        # the slice has to guarantee the objective delay for cbr and vbr if their traffics do not surpass the maximum
        cbr_fulfilled = cbr_th or cbr_prb or cbr_queue 
        vbr_fulfilled = vbr_th or vbr_prb or vbr_queue
        SLA_fulfilled = cbr_fulfilled and vbr_fulfilled
        return not(SLA_fulfilled)

    def get_state(self):
        '''converts the info into a normalized vector'''
        for i, var in enumerate(self.state_variables):
            self.state[i] = self.info[var] / self.norm_const[var]        
        return self.state


class SliceRANURLC(SliceRANeMBB):
    '''
    Ultra-Reliable Low-Latency Communications (URLLC) slice.
    Extends eMBB with tighter SLAs and different traffic parameters.
    '''
    def __init__(self, rng, user_counter, id, SLA, CBR_description, VBR_description, state_variables, norm_const, slots_per_step, slot_length = 1e-3):
        super().__init__(rng, user_counter, id, SLA, CBR_description, VBR_description, state_variables, norm_const, slots_per_step, slot_length)
        self.type = 'URLLC'

 #test for cordinate
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from numpy.random import default_rng
    
    # Test UE creation and attributes
    rng = default_rng(seed=42)
    ues = []
    for i in range(10):
        ue = UE(i, 0, CbrSource(bit_rate=1000), CBR)
        ue.x, ue.y = generate_xy(rng)
        ue.vx = rng.normal(0, 5)
        ue.vy = rng.normal(0, 5)
        ues.append(ue)
        print(f"UE {ue.id}: x={ue.x:.2f}, y={ue.y:.2f}, vx={ue.vx:.2f}, vy={ue.vy:.2f}")
    
    # Plot UE positions
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Hexagon boundary
    hex_x = [0, 0.25, 0.75, 1, 0.75, 0.25, 0]
    hex_y = [0.5, 1, 1, 0.5, 0, 0, 0.5]
    ax.plot(hex_x, hex_y, 'k-', linewidth=2, label='Cell Boundary')
    ax.fill(hex_x, hex_y, 'lightgray', alpha=0.3)
    
    # gNodeB
    ax.scatter(0.5, 0.5, c='red', s=200, marker='^', label='gNodeB', zorder=5)
    
    # UEs
    for ue in ues:
        ax.scatter(ue.x, ue.y, c='blue', s=50, alpha=0.7)
        ax.annotate(f'{ue.id}', (ue.x, ue.y), textcoords="offset points", xytext=(0,10), ha='center')
    
    ax.set_xlabel('X Coordinate (normalized)')
    ax.set_ylabel('Y Coordinate (normalized)')
    ax.set_title('UE Test: Positions and Velocities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.savefig('ue_test_plot.png', dpi=300, bbox_inches='tight')
    print("UE test plot saved as 'ue_test_plot.png'")
    plt.show()
