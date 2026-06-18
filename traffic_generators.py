#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

Classes:

PeriodicSource
OnOffSource
CbrSource
FixedPacketCbrSource
VbrSource

"""
import numpy as np

SLOT_LENGTH = 1e-3

class PeriodicSource:
    def __init__(self, packet_size = 640, period = 10):
        self.packet_size = packet_size
        self.period = period
        self.counter = self.period

    def step(self):
        self.counter = max(self.counter - 1, 0)
        if self.counter == 0:
            self.counter = self.period
            return self.packet_size
        else:
            return 0

class OnOffSource:
    def __init__(self, packet_size = 1000, period = 2, T_on = 500, T_off = 1000, initial_state = 1):
        self.T_on = T_on
        self.T_off = T_off
        self.state = initial_state
        self.periodic_source = PeriodicSource(packet_size, period)
        self.time_to_change = np.random.geometric(p = 1/T_off)

    def step(self):
        if self.time_to_change == 0:
            if self.state == 1:
                self.state = 0
                self.time_to_change = np.random.geometric(p = 1/self.T_on)
            else:
                self.state = 1
                self.time_to_change = np.random.geometric(p = 1/self.T_off)

        self.time_to_change = max(self.time_to_change - 1, 0)

        if self.state == 1:
            return self.periodic_source.step()
        else:
            return 0

class CbrSource(PeriodicSource):
    def __init__(self, bit_rate = 1000000, step_length = SLOT_LENGTH):
        packet_size = bit_rate * step_length
        super().__init__(packet_size = packet_size, period = 1)

class FixedPacketCbrSource:
    def __init__(
        self,
        packet_size = 12000,
        bit_rate = 1000000,
        step_length = SLOT_LENGTH,
        bit_rate_schedule = None,
    ):
        self.packet_size = float(packet_size)
        self.bit_rate = float(bit_rate)
        self.step_length = float(step_length)
        self.credit = 0.0
        self.step_counter = 0
        self.bit_rate_schedule = self._normalize_schedule(bit_rate_schedule)
        self._next_schedule_idx = 0

        if self.packet_size <= 0:
            raise ValueError("packet_size must be positive")
        if self.step_length <= 0:
            raise ValueError("step_length must be positive")

    def _normalize_schedule(self, schedule):
        if not schedule:
            return []

        normalized = []
        for item in schedule:
            if isinstance(item, dict):
                if "time_s" in item:
                    step = round(float(item["time_s"]) / self.step_length)
                elif "time" in item:
                    step = round(float(item["time"]) / self.step_length)
                elif "begin_s" in item:
                    step = round(float(item["begin_s"]) / self.step_length)
                else:
                    step = item.get("step", item.get("begin", 0))
                bit_rate = item["bit_rate"]
            else:
                step, bit_rate = item
            normalized.append((int(step), float(bit_rate)))

        return sorted(normalized, key=lambda item: item[0])

    def set_bit_rate(self, bit_rate):
        self.bit_rate = float(bit_rate)

    def step(self):
        while self._next_schedule_idx < len(self.bit_rate_schedule):
            step, bit_rate = self.bit_rate_schedule[self._next_schedule_idx]
            if self.step_counter < step:
                break
            self.bit_rate = bit_rate
            self._next_schedule_idx += 1

        self.credit += max(self.bit_rate, 0.0) * self.step_length
        packets = int(self.credit // self.packet_size)
        bits = packets * self.packet_size
        self.credit -= bits
        self.step_counter += 1
        return bits

class VbrSource:
    def __init__(self, packet_size = 1000, burst_size = 500, burst_rate = 1, step_length = SLOT_LENGTH):
        self.burst_size = burst_size
        self.packet_size = packet_size
        self.inter_arrival_steps = (1/burst_rate)/step_length
        self.steps_to_next_arrival = np.rint(np.random.exponential(self.inter_arrival_steps))
        self.active_bursts = []
        self.steps_to_go = []

    def step(self):
        bits = 0
        ending = []

        # active bursts
        for i, source in enumerate(self.active_bursts):
            if i >= len(self.steps_to_go):
                print(self.steps_to_go)
                print(self.active_bursts)
            self.steps_to_go[i] -= 1
            if self.steps_to_go[i] == 0:
                ending.append(i)
            else:
                bits += source.step()

        # ending bursts
        if len(ending) > 0:
            # self.steps_to_go = [steps for steps in self.steps_to_go if steps > 0]
            self.steps_to_go = [self.steps_to_go[i] for i, _ in enumerate(self.active_bursts) if i not in ending]
            self.active_bursts = [self.active_bursts[i] for i, _ in enumerate(self.active_bursts) if i not in ending]

        # arriving bursts
        self.steps_to_next_arrival -= 1
        if self.steps_to_next_arrival == 0:
            # new arrival
            self.active_bursts.append(PeriodicSource(packet_size = self.packet_size, period = 1))
            self.steps_to_go.append(np.rint(np.random.exponential(self.burst_size)))
            self.steps_to_next_arrival = np.rint(np.random.exponential(self.inter_arrival_steps))

        return bits

if __name__ == '__main__':
    source = VbrSource(burst_rate = 5)
    total_bits = 0
    for t in range(10000):
        bits = source.step()
        total_bits += bits
        if (t%100) == 0:
            print('VBR: t = {}: steps to next burst = {}, arriving bits = {}, total_bits = {}'.format(t, source.steps_to_next_arrival, bits, total_bits))
